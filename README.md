# Editing_Section_AI
The project intends to use object detection and image labeling to create an AI that can identify what part of a footage should be edited. The AI intends to help newcomers understand how to edit. 

#Introduction
A key challenge in Machine Learning (ML) is to build agents that can identify complex human environments in response to spoken or written commands.
Today's models, including robots, are often able to explore complex environments, 
It is not yet possible to understand a series of behavioral expressions expressed in natural language such as  “the main character walks past the door quickly and approaches the person wearing glasses. After looking directly at his face for a moment, he looks confident and delivers the yellow envelope he brought.”

This challenge, called Visual and Language Navigation (VLN). requires a sophisticated understanding of spatial language.
This is a study of video editing technology using artificial intelligence. Video production is a series of sequential processes, and the part that requires the most time is later work. 
The editing process so far has had to be reviewed again all the footage taken and elaborated on distinguishing between necessary and unnecessary parts based on scenarios and conti. In particular, this part is very important in editing, which is the post production stage that produces the final results, so the director or producer will participate in editing directly. 
The necessary techniques to solve these problems require techniques to understand both images and languages at the same time. Natural language processing technology and image recognition processing technology have been studied separately so far, so it was necessary to have a technology to understand the verbal expression in the video at a contextual level. 
Against this backdrop, this study was conducted to confirm the applicability of visual and linguistic techniques to image editing.

#AI'S FUNCTION
Open CV
Open CV is the main computer program used to create the AI for this project. 
It was used for its function to detect and lable the objects within the frame. 

I. Object Detection
Objects in the footages will be detected using YOLO:Real Time-Object Detection coding.
Each objects will be labeled with a certain tag; ie. cars, phones, robots, etc.

II. Image Labeling
First, the keywords and topic words of the script will be extracted using NLP code.
The number of keywords that needs to be extracted can be cut down by only extracting the nouns. 

Next, the keywords will be matched with the objects of the footage.
With the labeled objects from the previous coding, the keywords from the script can be matched with the objects from the frame. 

The time frame created by the matching frames will indicate where the editing section should be.

#Limitation
Due to the high cost of image labeling, a free program called cocodata for this project. 
Only 80 objects was able to be labeled. 
The limiting number of objects that can be labeled meant that the process of identifying the editing sections was hindered.





