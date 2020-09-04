from tkinter import *
import tkinter.messagebox as msgbox
from PIL import ImageTk, Image
from tkinter import ttk
import function_create
import Processing

# 참고: 파이썬 코딩 무료 강의 (활용편2) - GUI 프로그래밍을 배우고 '여러 이미지 합치기' 프로그램을 함께 만들어요. [나도코딩]
root = Tk()
variable = StringVar()
count_variable = StringVar()


def search_btn_click():
    global path
    global files
    path = function_create.add_dirctory_path()
    change(path)
    files = function_create.read_all_files(path)
    count_variable.set("전체 갯수 : "+str(len(files)))
    for file in files:
        list_file.insert(END, file)


def change(text):
    path_label.config(text=text)


def change_current_file_name(file, count):
    current_file_name = file
    variable.set(current_file_name)
    count_variable.set("전체 갯수 : "+str(count)+"/"+str(len(files)))
    file_name_frame.update()


def processing_btn():
    count = 0
    global processing_fin_image
    if path is not None:
        for file in files:
            count += 1
            change_current_file_name(file, count)
            processing_fin_image = Processing.processing_init(path, file)
            img = Image.fromarray(processing_fin_image[0])
            ori_img = Image.fromarray(processing_fin_image[1])
            roi_img = Image.fromarray(processing_fin_image[2])
            mask_img = Image.fromarray(processing_fin_image[3])

            original_img = ImageTk.PhotoImage(image=ori_img)
            original_image.config(image=original_img)
            original_image.image = original_img

            roi_img = ImageTk.PhotoImage(image=roi_img)
            roi_image.config(image=roi_img)
            roi_image.image = roi_img

            mask_img = ImageTk.PhotoImage(image=mask_img)
            mask_image.config(image=mask_img)
            mask_image.image = mask_img

            # 결과이미지
            result_img = ImageTk.PhotoImage(image=img)
            result_image.config(image=result_img)
            result_image.image = result_img
            percent = int((count/(len(files)))*100)
            p_var.set(percent)
            progress_bar.update
    else:
        # FIXME:경로없으면 아래 경고창 띄워야함
        msgbox.showinfo("경고", "폴더 경로 입력좀...")
    pass


if __name__ == '__main__':

    root.title("파이프 프로그램")
    root.geometry("750x1020")

    file_frame = Frame(root)
    file_frame.pack(fill="x")

    list_frame = Frame(root)
    list_frame.pack(fill="x")

    scrollbar = Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    list_file = Listbox(list_frame, selectmode="extended",
                        height=5, yscrollcommand=scrollbar.set)
    list_file.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=list_file.yview)

    search_btn = Button(file_frame, padx=5, pady=5, width=10,
                        text="찾기", command=search_btn_click)
    start_btn = Button(file_frame, padx=5, pady=5, width=10,
                       text="이미지처리 시작", command=processing_btn)

    path_label = Label(file_frame, text="경로를 입력 ㄱㄱ")
    path_label.pack(side="left")
    start_btn.pack(side="right")
    search_btn.pack(side="right")

    file_name_frame = LabelFrame(root, text="파일 이름")
    file_name_frame.pack(fill="x")

    txt_dest_path = Label(file_name_frame, textvariable=variable)
    txt_dest_path.pack(side="left", fill="x", expand=True)

    total_count_label = Label(file_name_frame, textvariable=count_variable)
    total_count_label.pack(side="right")

    list_image_frame = Frame(root)
    list_image_frame.pack(fill="x")

    original_img = ImageTk.PhotoImage(Image.open(
        "/home/hgh/hgh/부산프로젝트연구 opencv 코드/github용/pipe_program/q.png"))
    original_image = Label(list_image_frame, width=250,
                           height=250, image=original_img)
    original_image.pack(side="left", fill="x")

    roi_img = ImageTk.PhotoImage(Image.open(
        "/home/hgh/hgh/부산프로젝트연구 opencv 코드/github용/pipe_program/q.png"))
    roi_image = Label(list_image_frame, width=250, height=250, image=roi_img)
    roi_image.pack(side="left")

    mask_img = ImageTk.PhotoImage(Image.open(
        "/home/hgh/hgh/부산프로젝트연구 opencv 코드/github용/pipe_program/q.png"))
    mask_image = Label(list_image_frame, width=250, height=250, image=mask_img)
    mask_image.pack(side="left")

    result_img = ImageTk.PhotoImage(Image.open(
        "/home/hgh/hgh/부산프로젝트연구 opencv 코드/github용/pipe_program/q.png"))
    result_image = Label(root, width=736, height=480, image=result_img)
    result_image.pack(fill="x")

    next_prev_frame = Frame(root)
    next_prev_frame.pack(fill="x")
    prev_btn = Button(next_prev_frame, padx=5, pady=5, width=10,
                      text="다음", command=search_btn_click)
    next_btn = Button(next_prev_frame, padx=5, pady=5, width=10,
                      text="이전", command=search_btn_click)
    prev_btn.pack(side="right")
    next_btn.pack(side="right")

    frame_progress = LabelFrame(root, text="진행상황")
    frame_progress.pack(fill="x")

    p_var = DoubleVar()
    progress_bar = ttk.Progressbar(frame_progress, maximum=100, variable=p_var)
    progress_bar.pack(fill="x")

    root.resizable(False, False)
    root.mainloop()
