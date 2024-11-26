# app.py

import gradio as gr
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import time
import re
import base64
import logging
import os
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

# Import OpenAI library
import openai

# Suppress only the single warning from urllib3 needed.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging to output to the console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Initialize variables and models
logger.info("Initializing variables and models")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = None
bookmarks = []
fetch_cache = {}

# Lock for thread-safe operations
lock = threading.Lock()

# Define the categories
CATEGORIES = [
    "Social Media",
    "News and Media",
    "Education and Learning",
    "Entertainment",
    "Shopping and E-commerce",
    "Finance and Banking",
    "Technology",
    "Health and Fitness",
    "Travel and Tourism",
    "Food and Recipes",
    "Sports",
    "Arts and Culture",
    "Government and Politics",
    "Business and Economy",
    "Science and Research",
    "Personal Blogs and Journals",
    "Job Search and Careers",
    "Music and Audio",
    "Videos and Movies",
    "Reference and Knowledge Bases",
    "Dead Link",
    "Uncategorized",
]

# Set up Groq Cloud API key and base URL
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set.")

openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

# Initialize global variables for rate limiting
api_lock = threading.Lock()
last_api_call_time = 0

def extract_main_content(soup):
    """
    Extract the main content from a webpage while filtering out boilerplate content.
    """
    if not soup:
        return ""

    # Remove unwanted elements
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'noscript']):
        element.decompose()

    # Extract text from <p> tags
    p_tags = soup.find_all('p')
    if p_tags:
        content = ' '.join([p.get_text(strip=True, separator=' ') for p in p_tags])
    else:
        # Fallback to body content
        content = soup.get_text(separator=' ', strip=True)

    # Clean up the text
    content = re.sub(r'\s+', ' ', content)

    # Truncate content to a reasonable length (e.g., 1500 words)
    words = content.split()
    if len(words) > 1500:
        content = ' '.join(words[:1500])

    return content

def get_page_metadata(soup):
    """
    Extract metadata from the webpage including title, description, and keywords.
    """
    metadata = {
        'title': '',
        'description': '',
        'keywords': ''
    }

    if not soup:
        return metadata

    # Get title
    title_tag = soup.find('title')
    if title_tag and title_tag.string:
        metadata['title'] = title_tag.string.strip()

    # Get meta description
    meta_desc = (
        soup.find('meta', attrs={'name': 'description'}) or
        soup.find('meta', attrs={'property': 'og:description'}) or
        soup.find('meta', attrs={'name': 'twitter:description'})
    )
    if meta_desc:
        metadata['description'] = meta_desc.get('content', '').strip()

    # Get meta keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords:
        metadata['keywords'] = meta_keywords.get('content', '').strip()

    # Get OG title if main title is empty
    if not metadata['title']:
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            metadata['title'] = og_title.get('content', '').strip()

    return metadata
def generate_summary_and_assign_category(bookmark):
    """
    Generate a concise summary and assign a category using a single LLM call.
    """
    logger.info(f"Generating summary and assigning category for bookmark: {bookmark.get('url')}")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Rate Limiting Logic
            with api_lock:
                global last_api_call_time
                current_time = time.time()
                elapsed = current_time - last_api_call_time
                if elapsed < 2:
                    sleep_duration = 2 - elapsed
                    logger.info(f"Sleeping for {sleep_duration:.2f} seconds to respect rate limits.")
                    time.sleep(sleep_duration)
                last_api_call_time = time.time()

            html_content = bookmark.get('html_content', '')
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = get_page_metadata(soup)
            main_content = extract_main_content(soup)

            # Prepare content for the prompt
            content_parts = []
            if metadata['title']:
                content_parts.append(f"Title: {metadata['title']}")
            if metadata['description']:
                content_parts.append(f"Description: {metadata['description']}")
            if metadata['keywords']:
                content_parts.append(f"Keywords: {metadata['keywords']}")
            if main_content:
                content_parts.append(f"Main Content: {main_content}")

            content_text = '\n'.join(content_parts)

            # Detect insufficient or erroneous content
            error_keywords = ['Access Denied', 'Security Check', 'Cloudflare', 'captcha', 'unusual traffic']
            if not content_text or len(content_text.split()) < 50:
                use_prior_knowledge = True
                logger.info(f"Content for {bookmark.get('url')} is insufficient. Instructing LLM to use prior knowledge.")
            elif any(keyword.lower() in content_text.lower() for keyword in error_keywords):
                use_prior_knowledge = True
                logger.info(f"Content for {bookmark.get('url')} contains error messages. Instructing LLM to use prior knowledge.")
            else:
                use_prior_knowledge = False

            if use_prior_knowledge:
                prompt = f"""
You are a knowledgeable assistant with up-to-date information as of 2023.
URL: {bookmark.get('url')}
Provide:
1. A concise summary (max two sentences) about this website.
2. Assign the most appropriate category from the list below.
Categories:
{', '.join([f'"{cat}"' for cat in CATEGORIES])}
Format:
Summary: [Your summary]
Category: [One category]
"""
            else:
                prompt = f"""
You are an assistant that creates concise webpage summaries and assigns categories.
Content:
{content_text}
Provide:
1. A concise summary (max two sentences) focusing on the main topic.
2. Assign the most appropriate category from the list below.
Categories:
{', '.join([f'"{cat}"' for cat in CATEGORIES])}
Format:
Summary: [Your summary]
Category: [One category]
"""

            def estimate_tokens(text):
                return len(text) / 4

            prompt_tokens = estimate_tokens(prompt)
            max_tokens = 150
            total_tokens = prompt_tokens + max_tokens

            tokens_per_minute = 40000
            tokens_per_second = tokens_per_minute / 60
            required_delay = total_tokens / tokens_per_second
            sleep_time = max(required_delay, 2)

            response = openai.ChatCompletion.create(
                model='llama-3.1-70b-versatile',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(max_tokens),
                temperature=0.5,
            )

            content = response['choices'][0]['message']['content'].strip()
            if not content:
                raise ValueError("Empty response received from the model.")

            summary_match = re.search(r"Summary:\s*(.*)", content)
            category_match = re.search(r"Category:\s*(.*)", content)

            if summary_match:
                bookmark['summary'] = summary_match.group(1).strip()
            else:
                bookmark['summary'] = 'No summary available.'

            if category_match:
                category = category_match.group(1).strip().strip('"')
                if category in CATEGORIES:
                    bookmark['category'] = category
                else:
                    bookmark['category'] = 'Uncategorized'
            else:
                bookmark['category'] = 'Uncategorized'

            # Simple keyword-based validation
            summary_lower = bookmark['summary'].lower()
            url_lower = bookmark['url'].lower()
            if 'social media' in summary_lower or 'twitter' in summary_lower or 'x.com' in url_lower:
                bookmark['category'] = 'Social Media'
            elif 'wikipedia' in url_lower:
                bookmark['category'] = 'Reference and Knowledge Bases'

            logger.info("Successfully generated summary and assigned category")
            time.sleep(sleep_time)
            break

        except openai.error.RateLimitError as e:
            retry_count += 1
            wait_time = int(e.headers.get("Retry-After", 5))
            logger.warning(f"Rate limit reached. Waiting for {wait_time} seconds before retrying... (Attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Error generating summary and assigning category: {e}", exc_info=True)
            bookmark['summary'] = 'No summary available.'
            bookmark['category'] = 'Uncategorized'
            break

def parse_bookmarks(file_content):
    """
    Parse bookmarks from HTML file.
    """
    logger.info("Parsing bookmarks")
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        extracted_bookmarks = []
        for link in soup.find_all('a'):
            url = link.get('href')
            title = link.text.strip()
            if url and title:
                if url.startswith('http://') or url.startswith('https://'):
                    extracted_bookmarks.append({'url': url, 'title': title})
                else:
                    logger.info(f"Skipping non-http/https URL: {url}")
        logger.info(f"Extracted {len(extracted_bookmarks)} bookmarks")
        return extracted_bookmarks
    except Exception as e:
        logger.error("Error parsing bookmarks: %s", e, exc_info=True)
        raise

def fetch_url_info(bookmark):
    """
    Fetch information about a URL.
    """
    url = bookmark['url']
    if url in fetch_cache:
        with lock:
            bookmark.update(fetch_cache[url])
        return

    try:
        logger.info(f"Fetching URL info for: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=5, verify=False, allow_redirects=True)
        bookmark['etag'] = response.headers.get('ETag', 'N/A')
        bookmark['status_code'] = response.status_code

        content = response.text
        logger.info(f"Fetched content length for {url}: {len(content)} characters")

        if response.status_code >= 500:
            bookmark['dead_link'] = True
            bookmark['description'] = ''
            bookmark['html_content'] = ''
            logger.warning(f"Dead link detected: {url} with status {response.status_code}")
        else:
            bookmark['dead_link'] = False
            bookmark['html_content'] = content
            bookmark['description'] = ''
            logger.info(f"Fetched information for {url}")

    except requests.exceptions.Timeout:
        bookmark['dead_link'] = False
        bookmark['etag'] = 'N/A'
        bookmark['status_code'] = 'Timeout'
        bookmark['description'] = ''
        bookmark['html_content'] = ''
        bookmark['slow_link'] = True
        logger.warning(f"Timeout while fetching {url}. Marking as 'Slow'.")
    except Exception as e:
        bookmark['dead_link'] = True
        bookmark['etag'] = 'N/A'
        bookmark['status_code'] = 'Error'
        bookmark['description'] = ''
        bookmark['html_content'] = ''
        logger.error(f"Error fetching URL info for {url}: {e}", exc_info=True)
    finally:
        with lock:
            fetch_cache[url] = {
                'etag': bookmark.get('etag'),
                'status_code': bookmark.get('status_code'),
                'dead_link': bookmark.get('dead_link'),
                'description': bookmark.get('description'),
                'html_content': bookmark.get('html_content', ''),
                'slow_link': bookmark.get('slow_link', False),
            }

def vectorize_and_index(bookmarks_list):
    """
    Create vector embeddings for bookmarks and build FAISS index with ID mapping.
    """
    global faiss_index
    logger.info("Vectorizing summaries and building FAISS index")
    try:
        summaries = [bookmark['summary'] for bookmark in bookmarks_list]
        embeddings = embedding_model.encode(summaries)
        dimension = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        ids = np.array([bookmark['id'] for bookmark in bookmarks_list], dtype=np.int64)
        index.add_with_ids(np.array(embeddings).astype('float32'), ids)
        faiss_index = index
        logger.info("FAISS index built successfully with IDs")
        return index
    except Exception as e:
        logger.error(f"Error in vectorizing and indexing: {e}", exc_info=True)
        raise

def display_bookmarks():
    """
    Generate HTML display for bookmarks.
    """
    logger.info("Generating HTML display for bookmarks")
    cards = ''
    for i, bookmark in enumerate(bookmarks):
        index = i + 1
        if bookmark.get('dead_link'):
            status = "❌ Dead Link"
            card_style = "border: 2px solid red;"
            text_style = "color: white;"
        elif bookmark.get('slow_link'):
            status = "⏳ Slow Response"
            card_style = "border: 2px solid orange;"
            text_style = "color: white;"
        else:
            status = "✅ Active"
            card_style = "border: 2px solid green;"
            text_style = "color: white;"

        title = bookmark['title']
        url = bookmark['url']
        etag = bookmark.get('etag', 'N/A')
        summary = bookmark.get('summary', '')
        category = bookmark.get('category', 'Uncategorized')

        # Escape HTML content to prevent XSS attacks
        from html import escape
        title = escape(title)
        url = escape(url)
        summary = escape(summary)
        category = escape(category)

        card_html = f'''
        <div class="card" style="{card_style} padding: 10px; margin: 10px; border-radius: 5px; background-color: #1e1e1e;">
            <div class="card-content">
                <h3 style="{text_style}">{index}. {title} {status}</h3>
                <p style="{text_style}"><strong>Category:</strong> {category}</p>
                <p style="{text_style}"><strong>URL:</strong> <a href="{url}" target="_blank" style="{text_style}">{url}</a></p>
                <p style="{text_style}"><strong>ETag:</strong> {etag}</p>
                <p style="{text_style}"><strong>Summary:</strong> {summary}</p>
            </div>
        </div>
        '''
        cards += card_html
    logger.info("HTML display generated")
    return cards

def process_uploaded_file(file, state_bookmarks):
    """
    Process the uploaded bookmarks file.
    """
    global bookmarks, faiss_index
    logger.info("Processing uploaded file")

    if file is None:
        logger.warning("No file uploaded")
        return "Please upload a bookmarks HTML file.", '', state_bookmarks, display_bookmarks(), gr.update(choices=[])

    try:
        file_content = file.decode('utf-8')
    except UnicodeDecodeError as e:
        logger.error(f"Error decoding the file: {e}", exc_info=True)
        return "Error decoding the file. Please ensure it's a valid HTML file.", '', state_bookmarks, display_bookmarks(), gr.update(choices=[])

    try:
        bookmarks = parse_bookmarks(file_content)
    except Exception as e:
        logger.error(f"Error parsing bookmarks: {e}", exc_info=True)
        return "Error parsing the bookmarks HTML file.", '', state_bookmarks, display_bookmarks(), gr.update(choices=[])

    if not bookmarks:
        logger.warning("No bookmarks found in the uploaded file")
        return "No bookmarks found in the uploaded file.", '', state_bookmarks, display_bookmarks(), gr.update(choices=[])

    # Assign unique IDs to bookmarks
    for idx, bookmark in enumerate(bookmarks):
        bookmark['id'] = idx

    # Fetch bookmark info concurrently
    logger.info("Fetching URL info concurrently")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_url_info, bookmarks)

    # Process bookmarks concurrently with LLM calls
    logger.info("Processing bookmarks with LLM concurrently")
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(generate_summary_and_assign_category, bookmarks)

    try:
        faiss_index = vectorize_and_index(bookmarks)
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        return "Error building search index.", '', state_bookmarks, display_bookmarks(), gr.update(choices=[])

    message = f"✅ Successfully processed {len(bookmarks)} bookmarks."
    logger.info(message)

    # Generate displays and updates
    bookmark_html = display_bookmarks()
    choices = [f"{i+1}. {bookmark['title']} (Category: {bookmark['category']})"
               for i, bookmark in enumerate(bookmarks)]

    # Update state
    state_bookmarks = bookmarks.copy()

    return message, bookmark_html, state_bookmarks, bookmark_html, gr.update(choices=choices)

def delete_selected_bookmarks(selected_indices, state_bookmarks):
    """
    Delete selected bookmarks and remove their vectors from the FAISS index.
    """
    global bookmarks, faiss_index
    if not selected_indices:
        return "⚠️ No bookmarks selected.", gr.update(choices=[]), display_bookmarks()

    ids_to_delete = []
    indices_to_delete = []
    for s in selected_indices:
        idx = int(s.split('.')[0]) - 1
        if 0 <= idx < len(bookmarks):
            bookmark_id = bookmarks[idx]['id']
            ids_to_delete.append(bookmark_id)
            indices_to_delete.append(idx)
            logger.info(f"Deleting bookmark at index {idx + 1}")

    # Remove vectors from FAISS index
    if faiss_index is not None and ids_to_delete:
        faiss_index.remove_ids(np.array(ids_to_delete, dtype=np.int64))

    # Remove bookmarks from the list (reverse order to avoid index shifting)
    for idx in sorted(indices_to_delete, reverse=True):
        bookmarks.pop(idx)

    message = "🗑️ Selected bookmarks deleted successfully."
    logger.info(message)
    choices = [f"{i+1}. {bookmark['title']} (Category: {bookmark['category']})"
               for i, bookmark in enumerate(bookmarks)]

    # Update state
    state_bookmarks = bookmarks.copy()

    return message, gr.update(choices=choices), display_bookmarks()

def edit_selected_bookmarks_category(selected_indices, new_category, state_bookmarks):
    """
    Edit category of selected bookmarks.
    """
    if not selected_indices:
        return "⚠️ No bookmarks selected.", gr.update(choices=[]), display_bookmarks(), state_bookmarks
    if not new_category:
        return "⚠️ No new category selected.", gr.update(choices=[]), display_bookmarks(), state_bookmarks

    indices = [int(s.split('.')[0])-1 for s in selected_indices]
    for idx in indices:
        if 0 <= idx < len(bookmarks):
            bookmarks[idx]['category'] = new_category
            logger.info(f"Updated category for bookmark {idx + 1} to {new_category}")

    message = "✏️ Category updated for selected bookmarks."
    logger.info(message)

    # Update choices and display
    choices = [f"{i+1}. {bookmark['title']} (Category: {bookmark['category']})"
               for i, bookmark in enumerate(bookmarks)]

    # Update state
    state_bookmarks = bookmarks.copy()

    return message, gr.update(choices=choices), display_bookmarks(), state_bookmarks

def export_bookmarks():
    """
    Export bookmarks to an HTML file.
    """
    if not bookmarks:
        logger.warning("No bookmarks to export")
        return None

    try:
        logger.info("Exporting bookmarks to HTML")
        soup = BeautifulSoup("<!DOCTYPE NETSCAPE-Bookmark-file-1><Title>Bookmarks</Title><H1>Bookmarks</H1>", 'html.parser')
        dl = soup.new_tag('DL')
        for bookmark in bookmarks:
            dt = soup.new_tag('DT')
            a = soup.new_tag('A', href=bookmark['url'])
            a.string = bookmark['title']
            dt.append(a)
            dl.append(dt)
        soup.append(dl)
        html_content = str(soup)
        output_file = "exported_bookmarks.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("Bookmarks exported successfully")
        return output_file
    except Exception as e:
        logger.error(f"Error exporting bookmarks: {e}", exc_info=True)
        return None

def chatbot_response(user_query, chat_history):
    """
    Generate chatbot response using the FAISS index and embeddings.
    """
    if not bookmarks or faiss_index is None:
        logger.warning("No bookmarks available for chatbot")
        chat_history.append({"role": "assistant", "content": "⚠️ No bookmarks available. Please upload and process your bookmarks first."})
        return chat_history

    logger.info(f"Chatbot received query: {user_query}")

    try:
        chat_history.append({"role": "user", "content": user_query})

        with api_lock:
            global last_api_call_time
            current_time = time.time()
            elapsed = current_time - last_api_call_time
            if elapsed < 2:
                sleep_duration = 2 - elapsed
                logger.info(f"Sleeping for {sleep_duration:.2f} seconds to respect rate limits.")
                time.sleep(sleep_duration)
            last_api_call_time = time.time()

        query_vector = embedding_model.encode([user_query]).astype('float32')
        k = 5
        distances, ids = faiss_index.search(query_vector, k)
        ids = ids.flatten()

        id_to_bookmark = {bookmark['id']: bookmark for bookmark in bookmarks}
        matching_bookmarks = [id_to_bookmark.get(id) for id in ids if id in id_to_bookmark]

        if not matching_bookmarks:
            answer = "No relevant bookmarks found for your query."
            chat_history.append({"role": "assistant", "content": answer})
            return chat_history

        bookmarks_info = "\n".join([
            f"Title: {bookmark['title']}\nURL: {bookmark['url']}\nSummary: {bookmark['summary']}"
            for bookmark in matching_bookmarks
        ])

        prompt = f"""
A user asked: "{user_query}"
Based on the bookmarks below, provide a helpful answer to the user's query, referencing the relevant bookmarks.
Bookmarks:
{bookmarks_info}
Provide a concise and helpful response.
"""

        def estimate_tokens(text):
            return len(text) / 4

        prompt_tokens = estimate_tokens(prompt)
        max_tokens = 300
        total_tokens = prompt_tokens + max_tokens

        tokens_per_minute = 40000
        tokens_per_second = tokens_per_minute / 60
        required_delay = total_tokens / tokens_per_second
        sleep_time = max(required_delay, 2)

        response = openai.ChatCompletion.create(
            model='llama-3.1-70b-versatile',
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(max_tokens),
            temperature=0.7,
        )

        answer = response['choices'][0]['message']['content'].strip()
        logger.info("Chatbot response generated")
        time.sleep(sleep_time)

        chat_history.append({"role": "assistant", "content": answer})
        return chat_history

    except openai.error.RateLimitError as e:
        wait_time = int(e.headers.get("Retry-After", 5))
        logger.warning(f"Rate limit reached. Waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        return chatbot_response(user_query, chat_history)
    except Exception as e:
        error_message = f"⚠️ Error processing your query: {str(e)}"
        logger.error(error_message, exc_info=True)
        chat_history.append({"role": "assistant", "content": error_message})
        return chat_history
def build_app():
    """
    Build and launch the Gradio app.
    """
    try:
        logger.info("Building Gradio app")
        with gr.Blocks(css="app.css") as demo:
            # Initialize state
            state_bookmarks = gr.State([])

            # General Overview
            gr.Markdown("""
# 📚 SmartMarks - AI Browser Bookmarks Manager

Welcome to **SmartMarks**, your intelligent assistant for managing browser bookmarks. SmartMarks leverages AI to help you organize, search, and interact with your bookmarks seamlessly.

---

## 🚀 **How to Use SmartMarks**

SmartMarks is divided into three main sections:

1. **📂 Upload and Process Bookmarks:** Import your existing bookmarks and let SmartMarks analyze and categorize them for you.
2. **💬 Chat with Bookmarks:** Interact with your bookmarks using natural language queries to find relevant links effortlessly.
3. **🛠️ Manage Bookmarks:** View, edit, delete, and export your bookmarks with ease.

Navigate through the tabs to explore each feature in detail.
""")

            # Upload and Process Bookmarks Tab
            with gr.Tab("Upload and Process Bookmarks"):
                gr.Markdown("""
## 📂 **Upload and Process Bookmarks**

### 📝 **Steps to Upload and Process:**

1. **Upload Bookmarks File:**
   - Click on the **"📁 Upload Bookmarks HTML File"** button.
   - Select your browser's exported bookmarks HTML file from your device.

2. **Process Bookmarks:**
   - After uploading, click on the **"⚙️ Process Bookmarks"** button.
   - SmartMarks will parse your bookmarks, fetch additional information, generate summaries, and categorize each link based on predefined categories.

3. **View Processed Bookmarks:**
   - Once processing is complete, your bookmarks will be displayed in an organized and visually appealing format below.
""")

                upload = gr.File(label="📁 Upload Bookmarks HTML File", type='binary')
                process_button = gr.Button("⚙️ Process Bookmarks")
                output_text = gr.Textbox(label="✅ Output", interactive=False)
                bookmark_display = gr.HTML(label="📄 Processed Bookmarks")

            # Chat with Bookmarks Tab
            with gr.Tab("Chat with Bookmarks"):
                gr.Markdown("""
## 💬 **Chat with Bookmarks**

### 🤖 **How to Interact:**

1. **Enter Your Query:**
   - In the **"✍️ Ask about your bookmarks"** textbox, type your question or keyword related to your bookmarks.

2. **Submit Your Query:**
   - Click the **"📨 Send"** button to submit your query.

3. **Receive AI-Driven Responses:**
   - SmartMarks will analyze your query and provide relevant bookmarks that match your request.

4. **View Chat History:**
   - All your queries and the corresponding AI responses are displayed in the chat history.
""")

                chatbot = gr.Chatbot(label="💬 Chat with SmartMarks", type='messages')
                user_input = gr.Textbox(
                    label="✍️ Ask about your bookmarks",
                    placeholder="e.g., Do I have any bookmarks about AI?"
                )
                chat_button = gr.Button("📨 Send")

                chat_button.click(
                    chatbot_response,
                    inputs=[user_input, chatbot],
                    outputs=chatbot
                )

            # Manage Bookmarks Tab
            with gr.Tab("Manage Bookmarks"):
                gr.Markdown("""
## 🛠️ **Manage Bookmarks**

### 🗂️ **Features:**

1. **View Bookmarks:**
   - All your processed bookmarks are displayed here with their respective categories and summaries.

2. **Select Bookmarks:**
   - Use the checkboxes next to each bookmark to select one, multiple, or all bookmarks you wish to manage.

3. **Delete Selected Bookmarks:**
   - After selecting the desired bookmarks, click the **"🗑️ Delete Selected"** button to remove them from your list.

4. **Edit Categories:**
   - Select the bookmarks you want to re-categorize.
   - Choose a new category from the dropdown menu labeled **"🆕 New Category"**.
   - Click the **"✏️ Edit Category"** button to update their categories.

5. **Export Bookmarks:**
   - Click the **"💾 Export"** button to download your updated bookmarks as an HTML file.

6. **Refresh Bookmarks:**
   - Click the **"🔄 Refresh Bookmarks"** button to ensure the latest state is reflected in the display.
""")

                manage_output = gr.Textbox(label="🔄 Status", interactive=False)
                
                # Move bookmark_selector here
                bookmark_selector = gr.CheckboxGroup(
                    label="✅ Select Bookmarks",
                    choices=[]
                )
                
                new_category = gr.Dropdown(
                    label="🆕 New Category",
                    choices=CATEGORIES,
                    value="Uncategorized"
                )
                bookmark_display_manage = gr.HTML(label="📄 Bookmarks")

                with gr.Row():
                    delete_button = gr.Button("🗑️ Delete Selected")
                    edit_category_button = gr.Button("✏️ Edit Category")
                    export_button = gr.Button("💾 Export")
                    refresh_button = gr.Button("🔄 Refresh Bookmarks")

                download_link = gr.File(label="📥 Download Exported Bookmarks")

                # Connect all the button actions
                process_button.click(
                    process_uploaded_file,
                    inputs=[upload, state_bookmarks],
                    outputs=[output_text, bookmark_display, state_bookmarks, bookmark_display, bookmark_selector]
                )

                delete_button.click(
                    delete_selected_bookmarks,
                    inputs=[bookmark_selector, state_bookmarks],
                    outputs=[manage_output, bookmark_selector, bookmark_display_manage]
                )

                edit_category_button.click(
                    edit_selected_bookmarks_category,
                    inputs=[bookmark_selector, new_category, state_bookmarks],
                    outputs=[manage_output, bookmark_selector, bookmark_display_manage, state_bookmarks]
                )

                export_button.click(
                    export_bookmarks,
                    outputs=download_link
                )

                refresh_button.click(
                    lambda state_bookmarks: (
                        [
                            f"{i+1}. {bookmark['title']} (Category: {bookmark['category']})" 
                            for i, bookmark in enumerate(state_bookmarks)
                        ],
                        display_bookmarks()
                    ),
                    inputs=[state_bookmarks],
                    outputs=[bookmark_selector, bookmark_display_manage]
                )

        logger.info("Launching Gradio app")
        demo.launch(debug=True)
    except Exception as e:
        logger.error(f"Error building the app: {e}", exc_info=True)
        print(f"Error building the app: {e}")

if __name__ == "__main__":
    build_app()