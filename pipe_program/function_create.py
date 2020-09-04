from tkinter import filedialog
import os


def add_dirctory_path():
    directory = filedialog.askdirectory(initialdir="/home/hgh/hgh/busan_image")
    if directory is None:
        return "경로를 입력 ㄱㄱ"
    return directory


def read_all_files(directory):
    files_list = ""
    for root, dirs, files in os.walk(directory):
        files_list = files
    return files_list
