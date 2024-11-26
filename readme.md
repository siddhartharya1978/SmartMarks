---
title: Bookmark Manager
emoji: 😻
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---
📚 SmartMarks - AI Browser Bookmarks Manager


🌟 Overview
SmartMarks is your intelligent assistant for managing browser bookmarks. Leveraging the power of AI, SmartMarks helps you organize, search, and interact with your bookmarks seamlessly. Whether you're looking to categorize your links, retrieve information quickly, or maintain an updated list, SmartMarks has you covered.

Key Features
📂 Upload and Process Bookmarks: Import your existing bookmarks and let SmartMarks analyze and categorize them for you.
💬 Chat with Bookmarks: Interact with your bookmarks using natural language queries to find relevant links effortlessly.
🛠️ Manage Bookmarks: View, edit, delete, and export your bookmarks with ease.
🚀 High Processing Speed: Enhanced performance through concurrent fetching and processing of bookmarks.
🎨 Dynamic Color Theme: Automatically adapts to your system or browser's light/dark mode preferences for optimal readability and aesthetics.
🚀 How to Use SmartMarks
SmartMarks is divided into three main sections:

📂 Upload and Process Bookmarks: Import and process your bookmarks.
💬 Chat with Bookmarks: Ask questions about your bookmarks using natural language.
🛠️ Manage Bookmarks: Manage your bookmarks by viewing, editing, deleting, and exporting them.
Navigate through the tabs to explore each feature in detail.

📂 Upload and Process Bookmarks
📝 Steps:
Upload Bookmarks File:

Click on the "📁 Upload Bookmarks HTML File" button.
Select your browser's exported bookmarks HTML file from your device.
Process Bookmarks:

After uploading, click on the "⚙️ Process Bookmarks" button.
SmartMarks will parse your bookmarks, fetch additional information, generate summaries, and categorize each link based on predefined categories.
View Processed Bookmarks:

Once processing is complete, your bookmarks will be displayed in an organized and visually appealing format.
💬 Chat with Bookmarks
🤖 How to Interact:
Enter Your Query:

In the "✍️ Ask about your bookmarks" textbox, type your question or keyword related to your bookmarks. For example, "Do I have any bookmarks about AI?"
Submit Your Query:

Click the "📨 Send" button to submit your query.
Receive AI-Driven Responses:

SmartMarks will analyze your query and provide relevant bookmarks that match your request, making it easier to find specific links without manual searching.
View Chat History:

All your queries and the corresponding AI responses are displayed in the chat history for your reference.
🛠️ Manage Bookmarks
🗂️ Features:
View Bookmarks:

All your processed bookmarks are displayed here with their respective categories and summaries.
Select Bookmarks:

Use the checkboxes next to each bookmark to select one, multiple, or all bookmarks you wish to manage.
Delete Selected Bookmarks:

After selecting the desired bookmarks, click the "🗑️ Delete Selected" button to remove them from your list.
Edit Categories:

Select the bookmarks you want to re-categorize.
Choose a new category from the dropdown menu labeled "🆕 New Category".
Click the "✏️ Edit Category" button to update their categories.
Export Bookmarks:

Click the "💾 Export" button to download your updated bookmarks as an HTML file.
This file can be uploaded back to your browser to reflect the changes made within SmartMarks.
Refresh Bookmarks:

Click the "🔄 Refresh Bookmarks" button to ensure the latest state is reflected in the display.
📦 Deployment on Hugging Face Spaces
SmartMarks is designed to run seamlessly on Hugging Face Spaces, providing an interactive and user-friendly interface.

Prerequisites
Hugging Face Account: To deploy on Hugging Face Spaces, you need an account. Sign up here.
API Key: SmartMarks utilizes the Groq Cloud API for its chatbot functionality. Ensure you have a valid GROQ_API_KEY.
Repository Structure
markdown
Copy code
smartmarks/
├── app.py
├── app.css
├── requirements.txt
├── README.md

Setting Up Environment Variables
GROQ_API_KEY: SmartMarks requires the GROQ_API_KEY to interact with the Groq Cloud API. To set this up:

Navigate to your Space's dashboard on Hugging Face.

Go to Settings > Secrets.

Add a new secret with the key GROQ_API_KEY and paste your API key as the value.

Note: Keep your API keys secure. Never expose them in your code or repository.

Installing Dependencies
Hugging Face Spaces automatically installs dependencies listed in the requirements.txt file. Ensure your requirements.txt includes all necessary packages:

shell
Copy code
gradio
beautifulsoup4
sentence-transformers
faiss-cpu
requests
numpy
openai>=0.27.0,<1.0.0
Note: The uuid library is part of Python's standard library and doesn't require installation via pip. You can remove it from requirements.txt if it was previously included.

🌈 Dynamic Color Theme
SmartMarks automatically detects your system or browser's preferred color scheme and adjusts the application's theme accordingly:

Dark Mode: White text on a dark background for reduced eye strain in low-light environments.
Light Mode: Black text on a light background for bright environments.
These styles ensure that the text and background colors adapt based on the user's system or browser theme settings.

🔧 Configuration
Ensure that the GROQ_API_KEY environment variable is set correctly, as it's essential for the chatbot functionality to interact with the Groq Cloud API.

Setting Up Environment Variables
Navigate to Space Settings:

Go to your Space's dashboard on Hugging Face.
Add Secret:

Click on Settings > Secrets.

Add a new secret with the key GROQ_API_KEY and paste your API key as the value.

Note: Keep your API keys secure. Never expose them in your code or repository.

📋 Features in Detail
📂 Upload and Process Bookmarks
Upload Bookmarks File:

Click on the "📁 Upload Bookmarks HTML File" button.
Select your browser's exported bookmarks HTML file from your device.
Process Bookmarks:

After uploading, click on the "⚙️ Process Bookmarks" button.
SmartMarks will parse your bookmarks, fetch additional information, generate summaries, and categorize each link based on predefined categories.
View Processed Bookmarks:

Once processing is complete, your bookmarks will be displayed in an organized and visually appealing format.
💬 Chat with Bookmarks
Enter Your Query:

In the "✍️ Ask about your bookmarks" textbox, type your question or keyword related to your bookmarks. For example, "Do I have any bookmarks about AI?"
Submit Your Query:

Click the "📨 Send" button to submit your query.
Receive AI-Driven Responses:

SmartMarks will analyze your query and provide relevant bookmarks that match your request.
View Chat History:

All your queries and the corresponding AI responses are displayed in the chat history for your reference.
🛠️ Manage Bookmarks
View Bookmarks:

All your processed bookmarks are displayed here with their respective categories and summaries.
Select Bookmarks:

Use the checkboxes next to each bookmark to select one, multiple, or all bookmarks you wish to manage.
Delete Selected Bookmarks:

After selecting the desired bookmarks, click the "🗑️ Delete Selected" button to remove them from your list.
Edit Categories:

Select the bookmarks you want to re-categorize.
Choose a new category from the dropdown menu labeled "🆕 New Category".
Click the "✏️ Edit Category" button to update their categories.
Export Bookmarks:

Click the "💾 Export" button to download your updated bookmarks as an HTML file.
This file can be uploaded back to your browser to reflect the changes made within SmartMarks.
Refresh Bookmarks:

Click the "🔄 Refresh Bookmarks" button to ensure the latest state is reflected in the display.
🤝 Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the Repository:

Click on the "Fork" button at the top right of the repository page.
Clone Your Fork:

bash
Copy code
git clone https://github.com/your-username/smartmarks.git
cd smartmarks
Create a New Branch:

bash
Copy code
git checkout -b feature/YourFeature
Make Your Changes:

Implement your feature or bug fix.
Commit Your Changes:

bash
Copy code
git commit -m "Add some feature"
Push to Your Fork:

bash
Copy code
git push origin feature/YourFeature
Open a Pull Request:

Navigate to your forked repository on GitHub and click on "Compare & pull request".
📜 License
This project is licensed under the MIT License.

📞 Contact
For any questions or feedback, please contact siddhartharya@gmail.com

Happy Bookmarking with SmartMarks! 🚀📚

