# Creating Datasets

## Real world datasets

1. Take a high quality video going through a course.
2. Convert the video to .mp4 if necessary.
3. Store the video in DrobBox ".../ML/raw_videos".
4. Place the video in the UMARV-CV repo "/parapeters/input" directory
5. Run the script "/src/scripts/get_frames_from_video.py"
6. Go to https://app.roboflow.com/umarv-cv
7. Click "+ Create New Project"
![Alt text](visual_aids/create_datasets_1.png | width=100)
8. Fill in info like so, with Project name being "real_world/{name_of_dataset}"
![Alt text](visual_aids/create_datasets_2.png | width=100)
9. Drop all the raw images into roboflow by selecting the folder and pointing to "{UMARV-CV repo}/parameters/output/{dataset_name}/data"
10. Once the raw images are exported to roboflow, delete the contents of input and output in the repo.
11. Click Save and Continue
![Alt text](image.png | width=100)

## Unity datasets

...