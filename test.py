from tkinter import *

window = Tk()
text1 = "Waiting for Button Press..."
lbl = Label(window, text=text1)
lbl.pack()


def main():
    while True:
        if condition:
            '''The condition is GPIO READ statement but for simplicity I have used condition'''
            lbl["text"] = "IN Button Pressed.Loading Camera."


window.after(5000, main)
window.mainloop()
