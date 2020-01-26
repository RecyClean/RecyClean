# RecyClean

## Inspiration
McMaster's SRA presidential debate brought to light the issue of garbage sorting on campus. Many recycling bins were contaminated and were subsequently thrown into a landfill. During the project's development, we became aware of the many applications of this technology, including sorting raw materials, and manufacturing parts.

## What it does
The program takes a customizable trained deep learning model that can categorize over 1000 different classes of objects. When an object is placed in the foreground of the camera, its material is determined and its corresponding indicator light flashes. This is to replicate a small-scale automated sorting machine.   

## How we built it
To begin, we studied relevant modules of the OpenCV library and explored ways to implement them for our specific project. We also determined specific categories/materials for different classes of objects to build our own library for sorting.

## Challenges we ran into
Due to time constraints, we were unable to train our own data set for the specific objects we wanted. Many pre-trained models are designed to run on much stronger hardware than a raspberry pi. Being limited to pre-trained databases added a level of difficulty for the software to detect our specific objects.

## Accomplishments that we're proud of
The project actually worked and was surprisingly better than we had anticipated. We are proud that we were able to find a compromise in the pre-trained model and still have a functioning application.

## What we learned
We learned how to use OpenCV for this application, and the many applications of this technology in the deep learning and IoT industry.

## What's next for Smart Materials Sort
We'd love to find a way to dynamically update the training model (supervised learning), and try the software with our own custom models.
