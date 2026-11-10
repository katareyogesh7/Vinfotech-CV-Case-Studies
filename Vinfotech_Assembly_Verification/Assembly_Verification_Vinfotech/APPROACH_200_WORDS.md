# Case Study 1 — Approach write-up (under 200 words)

I built a CPU-only assembly verification prototype using Python, OpenCV and NumPy. It uses classical image processing rather than a trained model. Three known-good images establish reference scores and trial thresholds. Every test image passes through the same functions; verdicts are not based on filenames.

I locate the brass gear using HSV colour segmentation, morphology, contours and image moments. Its centre and size guide checks for the output nut, idler gear and compound gear. Colour masks also inspect the two circlips. I locate the blue base plate and straighten it using a perspective transform. This enables reusable inspection locations for corner bolts, bearing screws, motor mounting screws, the visible pinion, connector and cable. Grey coverage and Canny edge counts provide additional scores.

The program prints an OK/NOT OK verdict, flags suspicious parts and generates annotated PNGs and CSV reports. I chose these methods because they are explainable and CPU-friendly. Limitations include changes in lighting, resolution, camera angle and component placement. Bearing-block bodies and the hidden motor are not independently verified. Thresholds were tuned during development, so performance on the supplied six images is not an independent evaluation of unseen defects.
