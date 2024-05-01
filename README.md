## Artificial Facial Detection & Classification: Separating Real Faces from the Artificially Generated
#### Ante Tonkovic-Capin & Udhbhav Gupta

---
### Project Structure
This repository is a part of a broader submission for Computer Vision 766 at the University of Wisconsin-Madison for Spring 2024 semester. It represents the website and presentation code portion and is a companion repository for the related [source repository](https://github.com/AnteT/cs766-project) which contains the source code used for training, validation and testing.

The website is available [here](https://antet.github.io/ffd-app), while the rest of the project is structured as follows:

```text
Project
   ├── assets/
   ├── index.html
   ├── inference/
   │   ├── ffd.pt
   │   ├── fake-face.jpg
   │   ├── real-face.jpg
   │   ├── requirements.txt
   │   └── run_inference.py
   └── presentation-slides/
```

To run the model for inference, or reproduce the demonstration, follow the instructions below.

---

### Run Inference
Install the inference dependencies with `pip install -r requirements.txt`, then run the primary inference script `run_inference.py` using either the provided example images or your own. To run inference on your own images:

```bash
$ python ./run_inference.py [-d] "image1.png" ["image2.png"] ...
```

Multiple images can be provided and at least one is required for usage. Provide `-d, --display` flag to display image with the result. To reproduce the [demo](https://antet.github.io/ffd-app/#demonstration) referenced on the project website, use the `run_demo()` function or run:

```bash
$ python ./run_inference.py "./real-face.jpg" "./fake-face.jpg" --display 
```

Ensure the pretrained model `ffd.pt` and sample images, `real-face.jpg` and `fake-face.jpg` were correctly downloaded and are available in the project directory before running inference.

---

### Source Code Repository

The full project repository, source code, training results, prior models, training datasets, project proposals and midterm report are all available in the [project source repository](https://github.com/AnteT/cs766-project).

---

### Website and Demo Repository

The website repository, inference demos, sample images and presentation slides are all available in the [website and demo repository](https://github.com/AnteT/ffd-app). This is the repository you are currently in. The slides can be found in this repo in the `presentation-slides` directory:

```text
Project
   └── presentation-slides/
       └── Artificial Facial Detection.pptx
```

Questions or comments? Please don't hesitate to reach out!

---

**Thank you!**
_Ante Tonkovic-Capin & Udhbhav Gupta_