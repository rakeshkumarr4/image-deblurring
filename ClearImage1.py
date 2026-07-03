import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.restoration import richardson_lucy

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clear Image")
        self.root.geometry("1200x700")

        self.original_image = None
        self.deblurred_image = None
        self.photo_left = None
        self.photo_right = None

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill="x", padx=5, pady=5)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg="lightgray")
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.main_frame, bg="white")
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.left_label = tk.Label(self.left_frame, bg="lightgray")
        self.left_label.place(relx=0.5, rely=0.5, anchor="center")

        self.right_label = tk.Label(self.right_frame, bg="white")
        self.right_label.place(relx=0.5, rely=0.5, anchor="center")

        tk.Button(self.button_frame, text="Load Image", command=self.load_image).pack(side="left", padx=5)

        tk.Button(self.button_frame, text="Deblur Image", command=self.deblur_image).pack(side="left", padx=5)

        tk.Button(self.button_frame, text="Save Image", command=self.save_image).pack(side="left", padx=5)

        tk.Button(self.button_frame, text="Reset", command=self.reset_images).pack(side="left", padx=5)

    def display_image(self, image, label, frame, side):
        frame.update_idletasks()

        width = frame.winfo_width()
        height = frame.winfo_height()

        display = image.copy()
        display.thumbnail((width - 20, height - 20))

        photo = ImageTk.PhotoImage(display)

        label.config(image=photo)
        label.image = photo

        if side == "left":
            self.photo_left = photo
        else:
            self.photo_right = photo

    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not filepath:
            return

        self.original_image = Image.open(filepath).convert("RGB")

        self.display_image(
            self.original_image,
            self.left_label,
            self.left_frame,
            "left"
        )

    def deblur_image(self):
        if self.original_image is None:
            return

        img = np.array(self.original_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
        sharpen = cv2.filter2D(img, -1, sharpen_kernel)

        sharpen = sharpen.astype(np.float32) / 255.0

        #Choose a PSF (Point Spread Function) for deblurring

        # # Create a circular PSF (defocus kernel)
        # kernel_size = 27
        # radius = 9

        # psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        # cv2.circle(psf,
        #         (kernel_size // 2, kernel_size // 2),
        #         radius,
        #         1,
        #         -1)
        # psf /= psf.sum()
        # # (END) Create a circular PSF (defocus kernel)

        # Create a linear PSF (motion blur kernel)
        kernel_size = 13 #Must be odd
        angle = 135      # degrees
                        # 0°	Horizontal (left ↔ right)
                        # 30°	Slightly upward to the right
                        # 45°	Diagonal (bottom-left ↔ top-right)
                        # 60°	Steeper diagonal
                        # 90°	Vertical (up ↕ down)
                        # 120°	Steep diagonal (top-left ↔ bottom-right)
                        # 135°	Diagonal (top-left ↔ bottom-right)
                        # 150°	Slightly downward to the right
                        # 180°	Same as 0°
        psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.line(
            psf,
            (0, kernel_size // 2),
            (kernel_size - 1, kernel_size // 2),
            1,
            1
        )
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2),
            angle,
            1
        )
        psf = cv2.warpAffine(psf, M, (kernel_size, kernel_size))
        psf /= psf.sum()
        # (END) Create a linear PSF (motion blur kernel)

        # Richardson-Lucy on each color channel
        result = np.zeros_like(sharpen)

        for c in range(3):
            result[:, :, c] = richardson_lucy(
                sharpen[:, :, c],
                psf,
                num_iter=20
            )

        result = np.clip(result, 0, 1)

        # Convert back to uint8
        result = (result * 255).astype(np.uint8)

        # Optional denoising
        result = cv2.fastNlMeansDenoisingColored(
            result,
            None,
            10,
            10,
            7,
            21
        )

        result_pil = Image.fromarray(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        )

        self.deblurred_image = result_pil

        self.display_image(
            result_pil,
            self.right_label,
            self.right_frame,
            "right"
        )

    def save_image(self):
        if self.deblurred_image is None:
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp")
            ]
        )

        if filepath:
            self.deblurred_image.save(filepath)

    def reset_images(self):
        self.original_image = None
        self.deblurred_image = None
        self.photo_left = None
        self.photo_right = None

        self.left_label.config(image="")
        self.right_label.config(image="")

        self.left_label.image = None
        self.right_label.image = None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()