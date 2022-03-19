import tkinter as tk
import ctypes
import sys
#see https://stackoverflow.com/questions/57647034/prevent-sleep-mode-python-wakelock-on-python Devil answer

def display_on():
    print("Always On")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

def display_reset():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    sys.exit(0)


root = tk.Tk()
root.geometry("200x60")
root.title("Display App")
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame,
                   text="Quit",
                   fg="red",
                   command=display_reset)
button.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="Always ON",
                   command=display_on)
slogan.pack(side=tk.LEFT)

root.mainloop()
