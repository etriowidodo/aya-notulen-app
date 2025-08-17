import os
import shutil
import threading
import time
from datetime import datetime
from tkinter import *
from tkinter import filedialog, messagebox
from collections import defaultdict


class FileOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto File Organizer")
        self.root.geometry("500x400")

        # Variables
        self.folder_path = StringVar()
        self.running = False
        self.service_thread = None
        self.organize_option = IntVar(value=1)  # 1=Date, 2=Type
        self.file_map = defaultdict(list)
        self.folder_map = defaultdict(list)

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Folder Selection
        Label(self.root, text="Folder to Organize:").pack(pady=5)
        Entry(self.root, textvariable=self.folder_path, width=50).pack(pady=5)
        Button(self.root, text="Browse", command=self.browse_folder).pack(pady=5)

        # Organization Options
        Label(self.root, text="Organization Method:").pack(pady=10)
        Radiobutton(self.root, text="By Date (Year/Month/Day)", variable=self.organize_option, value=1).pack(anchor=W)
        Radiobutton(self.root, text="By File Type", variable=self.organize_option, value=2).pack(anchor=W)

        # Service Control
        Button(self.root, text="Start Service", command=self.start_service, bg="green", fg="white").pack(pady=15)
        Button(self.root, text="Stop Service", command=self.stop_service, bg="red", fg="white").pack(pady=5)

        # Status
        self.status_label = Label(self.root, text="Service: Stopped", fg="red")
        self.status_label.pack(pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def start_service(self):
        if not self.folder_path.get():
            messagebox.showerror("Error", "Please select a folder first")
            return

        if not self.running:
            self.running = True
            self.status_label.config(text="Service: Running", fg="green")
            self.service_thread = threading.Thread(target=self.run_service, daemon=True)
            self.service_thread.start()
            messagebox.showinfo("Info", "Background service started")

    def stop_service(self):
        if self.running:
            self.running = False
            self.status_label.config(text="Service: Stopped", fg="red")
            messagebox.showinfo("Info", "Background service stopped")

    def run_service(self):
        while self.running:
            try:
                folder = self.folder_path.get()
                if not os.path.exists(folder):
                    time.sleep(5)
                    continue

                # Clear previous mappings
                self.file_map.clear()
                self.folder_map.clear()

                # First pass: Scan and categorize all files
                self.scan_folder_contents(folder)

                # Second pass: Organize based on categorization
                self.organize_files()
                self.organize_folders()

                time.sleep(10)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)

    def scan_folder_contents(self, folder):
        """Scan all files and categorize them without moving"""
        for item in os.listdir(folder):
            if not self.running:
                break

            item_path = os.path.join(folder, item)

            # Skip if already in organized structure
            if self.is_in_organized_structure(item_path, folder):
                continue

            if os.path.isfile(item_path):
                if self.organize_option.get() == 1:
                    category = self.get_date_category(item_path)
                    self.file_map[category].append(item_path)
                else:
                    category = self.get_type_category(item_path)
                    self.file_map[category].append(item_path)
            elif os.path.isdir(item_path):
                if self.organize_option.get() == 1:
                    category = self.get_folder_date_category(item_path)
                    self.folder_map[category].append(item_path)
                else:
                    category = self.get_folder_type_category(item_path)
                    self.folder_map[category].append(item_path)

    def organize_files(self):
        """Move all files based on their categories"""
        for category, files in self.file_map.items():
            target_dir = os.path.join(self.folder_path.get(), category)
            os.makedirs(target_dir, exist_ok=True)

            for filepath in files:
                try:
                    if os.path.dirname(filepath) != target_dir:
                        shutil.move(filepath, os.path.join(target_dir, os.path.basename(filepath)))
                except Exception as e:
                    print(f"Error moving file {filepath}: {e}")

    def organize_folders(self):
        """Move all folders based on their categories"""
        for category, folders in self.folder_map.items():
            target_dir = os.path.join(self.folder_path.get(), category)
            os.makedirs(target_dir, exist_ok=True)

            for folderpath in folders:
                try:
                    if os.path.dirname(folderpath) != target_dir:
                        shutil.move(folderpath, os.path.join(target_dir, os.path.basename(folderpath)))
                except Exception as e:
                    print(f"Error moving folder {folderpath}: {e}")

    def is_in_organized_structure(self, item_path, base_folder):
        """Check if item is already in organized folder structure"""
        parent = os.path.dirname(item_path)
        if parent == base_folder:
            return False

        # Check if parent is one of our organized folders
        parent_name = os.path.basename(parent)
        organized_prefixes = ("20",)  # Years prefix
        return parent_name.startswith(organized_prefixes)

    def get_date_category(self, filepath):
        """Get date-based category for file"""
        timestamp = os.path.getmtime(filepath)
        date = datetime.fromtimestamp(timestamp)
        return os.path.join(date.strftime("%Y"), date.strftime("%m - %B"), date.strftime("%d"))

    def get_folder_date_category(self, folderpath):
        """Get date-based category for folder"""
        newest_file = self.get_newest_file(folderpath)
        if newest_file:
            timestamp = os.path.getmtime(newest_file)
        else:
            timestamp = os.path.getmtime(folderpath)

        date = datetime.fromtimestamp(timestamp)
        return os.path.join(date.strftime("%Y"), date.strftime("%m - %B"), date.strftime("%d"))

    def get_type_category(self, filepath):
        """Get type-based category for file"""
        ext = os.path.splitext(filepath)[1][1:].lower()

        if not ext:
            return "NO_EXT"

        # Group similar extensions
        if ext in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}:
            return "IMAGES"
        elif ext in {'doc', 'docx', 'txt', 'rtf'}:
            return "DOCUMENTS"
        elif ext in {'xls', 'xlsx', 'csv'}:
            return "SPREADSHEETS"
        elif ext == 'pdf':
            return "PDF"
        elif ext in {'mp3', 'wav', 'ogg'}:
            return "AUDIO"
        elif ext in {'mp4', 'avi', 'mkv', 'mov'}:
            return "VIDEO"
        elif ext in {'exe', 'msi'}:
            return "PROGRAMS"
        elif ext in {'zip', 'rar', '7z'}:
            return "ARCHIVES"
        elif ext in {'sql', 'db', 'sqlite'}:
            return "DATABASES"
        else:
            return ext.upper()

    def get_folder_type_category(self, folderpath):
        """Get type-based category for folder"""
        ext_counts = defaultdict(int)

        for root, _, files in os.walk(folderpath):
            for file in files:
                ext = os.path.splitext(file)[1][1:].lower()
                if ext:
                    ext_counts[ext] += 1

        if not ext_counts:
            return "EMPTY_FOLDERS"

        # Get most common extension
        most_common_ext = max(ext_counts, key=ext_counts.get)

        # Map to folder type
        if most_common_ext in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}:
            return "IMAGE_FOLDERS"
        elif most_common_ext in {'doc', 'docx', 'txt', 'rtf'}:
            return "DOCUMENT_FOLDERS"
        elif most_common_ext in {'xls', 'xlsx', 'csv'}:
            return "SPREADSHEET_FOLDERS"
        elif most_common_ext == 'pdf':
            return "PDF_FOLDERS"
        elif most_common_ext in {'mp3', 'wav', 'ogg'}:
            return "AUDIO_FOLDERS"
        elif most_common_ext in {'mp4', 'avi', 'mkv', 'mov'}:
            return "VIDEO_FOLDERS"
        elif most_common_ext in {'exe', 'msi'}:
            return "PROGRAM_FOLDERS"
        elif most_common_ext in {'zip', 'rar', '7z'}:
            return "ARCHIVE_FOLDERS"
        elif most_common_ext in {'sql', 'db', 'sqlite'}:
            return "DATABASE_FOLDERS"
        else:
            return f"{most_common_ext.upper()}_FOLDERS"

    def get_newest_file(self, folder):
        """Get the newest file in a folder"""
        newest_file = None
        newest_time = 0

        for root, _, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(root, file)
                file_time = os.path.getmtime(filepath)
                if file_time > newest_time:
                    newest_time = file_time
                    newest_file = filepath

        return newest_file


if __name__ == "__main__":
    root = Tk()
    app = FileOrganizerApp(root)
    root.mainloop()