import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import subprocess
import json

class VideoProcessor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Processing GUI")
        self.geometry("830x700")
        self.configure(bg="#f0f0f0") 

        self.video_path = ""
        self.output_path = ""
        self.excel_file_path = "result_matrix1.xlsx"

        self.create_widgets()

    def create_widgets(self):
        # Create a canvas to hold the frame
        self.canvas = tk.Canvas(self, bg="#f0f0f0")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")

        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)

        self.canvas.bind_all("<Up>", self.scroll_up)
        self.canvas.bind_all("<Down>", self.scroll_down)

        frame_style = {"padx": 10, "pady": 10, "bd": 2, "relief": tk.RIDGE, "bg": "#ffffff"}
        label_style = {"padx": 5, "pady": 5, "bg": "#f0f0f0", "font": ("Arial", 12)}
        entry_style = {"font": ("Arial", 12)}

        self.upload_btn = tk.Button(self.scrollable_frame, text="Upload Video", command=self.upload_video, font=("Arial", 12))
        self.upload_btn.pack(pady=20)

        self.video_label = tk.Label(self.scrollable_frame, bg="#f0f0f0")
        self.video_label.pack()

        row_col_frame = tk.Frame(self.scrollable_frame, **frame_style)
        row_col_frame.pack(pady=20)

        self.row_label = tk.Label(row_col_frame, text="Enter Number of Rows:", **label_style)
        self.row_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.row_entry = tk.Entry(row_col_frame, **entry_style)
        self.row_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.col_label = tk.Label(row_col_frame, text="Enter Number of Columns:", **label_style)
        self.col_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.col_entry = tk.Entry(row_col_frame, **entry_style)
        self.col_entry.pack(side=tk.LEFT, padx=5, pady=5)

        user_choice_frame = tk.Frame(self.scrollable_frame, **frame_style)
        user_choice_frame.pack(pady=20)

        self.user_choice_label = tk.Label(user_choice_frame, text="Color Channels:", **label_style)
        self.user_choice_label.grid(row=0, column=0, padx=5, pady=5)
        self.user_choice_entry = tk.Entry(user_choice_frame, **entry_style)
        self.user_choice_entry.grid(row=0, column=1, padx=5, pady=5)

        seq_par_frame = tk.Frame(self.scrollable_frame, **frame_style)
        seq_par_frame.pack(pady=20)

        self.seq_par_label = tk.Label(seq_par_frame, text="Select Sequential or Parallel:", **label_style)
        self.seq_par_label.grid(row=0, column=0, padx=5, pady=5)

        self.seq_par_var = tk.StringVar(value="Sequential")
        self.seq_radio = tk.Radiobutton(seq_par_frame, text="Sequential", variable=self.seq_par_var, value="Sequential", bg="#ffffff", font=("Arial", 12))
        self.seq_radio.grid(row=0, column=1, padx=5, pady=5)
        self.par_radio = tk.Radiobutton(seq_par_frame, text="Parallel", variable=self.seq_par_var, value="Parallel", bg="#ffffff", font=("Arial", 12))
        self.par_radio.grid(row=0, column=2, padx=5, pady=5)

        self.process_btn = tk.Button(self.scrollable_frame, text="Process Video", command=self.process_video, font=("Arial", 12))
        self.process_btn.pack(pady=20)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def scroll_up(self, event):
        self.canvas.yview_scroll(-1, "units")

    def scroll_down(self, event):
        self.canvas.yview_scroll(1, "units")

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            self.video_path = file_path
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.imgtk = imgtk
            cap.release()

    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first.")
            return

        try:
            rows = int(self.row_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of rows.")
            return

        try:
            cols = int(self.col_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of columns.")
            return

        user_choice = self.user_choice_entry.get()
        if not user_choice:
            messagebox.showerror("Error", "Please enter a valid user choice.")
            return

        seq_par_choice = self.seq_par_var.get()

        # Ask for the output file location
        # output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if True:
            
            # Store data in JSON file
            data = {
                "video": self.video_path,
                "color_channel": user_choice,
                "grids": {"rows": rows, "cols": cols},
                "execution_mode": seq_par_choice
            }

            json_path = "user_input_data.json"  # You can also ask the user where to save it
            with open(json_path, "w") as json_file:
                json.dump(data, json_file, indent=4)

            print(f"User input details saved to {json_path}")
            
            with open("user_input_data.json", "r") as file:
                config = json.load(file)

            rows, cols = config["grids"]["rows"], config["grids"]["cols"]
            execution_mode = config["execution_mode"]

            if execution_mode == "Parallel":
                command = ['python', 'rparallel_he3.py']
            else:
                command = ['python', 'seq.py']

            print(command)
            result =subprocess.run(command)
            if result.returncode == 0:
                messagebox.showinfo("Success", "Video processed successfully!")
                print("Output:", result.stdout)
            else:
                messagebox.showerror("Error", f"Processing failed:\n{result.stderr}")
                print("Error:", result.stderr)
            # self.run_processing_script(self.video_path, rows, cols, user_choice, seq_par_choice)
        else:
            messagebox.showerror("Error", "Please select a location to save the processed video.")

    def run_processing_script(self, video_path, rows, cols, user_choice, seq_par_choice):
        # Here we are calling the video processing script with the provided parameters.
        # Replace 'your_script.py' with the actual script name.
        script_name = 'INTERNSHIP/seq.py' if seq_par_choice == "Sequential" else 'INTERNSHIP/rparallel_he3.py'

        try:
            result = subprocess.run(['python', script_name, video_path, str(rows), str(cols), user_choice, self.output_path], capture_output=True, text=True)
            print("Result stdout:", result.stdout)  # Print stdout to debug
            print("Result stderr:", result.stderr)  # Print stderr to debug
            if result.returncode == 0:
                self.display_processed_video()
                messagebox.showinfo("Success", "Video Processed")
            else:
                messagebox.showerror("Error", f"An error occurred: {result.stderr}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_processed_video(self):
        cap = cv2.VideoCapture(self.output_path)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.imgtk = imgtk
                self.video_label.after(33, update_frame)  # 30 frames per second ~ 33ms per frame
            else:
                cap.release()

        update_frame()

if __name__ == "__main__":
    app = VideoProcessor()
    app.mainloop()
