# Hackathon-2021-Reindeer-Max-Team
By Reindeer Max Team

Hello everyone. We are three middle schoolers who love the computer and program. We are Reindeer Max team. Our project is focus detective system, which is an application to detect student’s distraction when they don’t pay attention to the class during online learning. 

The outbreak of Covid-19 has caused schools to embrace online learning due to lockdown and closure of campus. The continuous spread of virus has permanently changed education and teaching mode. We must be able to adapt to study online currently or in the future at any emergent situation. But the critical problem for E-learning is concentration on class, especially for primary or junior school students, because it is difficult for them to focus on class if there are no other people sitting around them. 

Project Description of Focus Detective System


Our application can help schools, teachers or parents to solve the problem of distraction of students when they have online class.
Below is project flow chart:

![image](https://user-images.githubusercontent.com/57511227/116418257-8c617c00-a86e-11eb-8ea8-feac6f32912e.png)

This software detects two major types of distraction: Fatigue and Off-job. Fatigue means the tiredness of students in front of screen. They will show yawn, nod-off for snooze, or frequent wink because of tiredness on their facial expression.  Off-job means the absence of students in front of screen. At the same time, we need to distinguish between the normal facial expression and the distraction expression. So, there is time boxes of fatigue and off-job are to input the minimum duration of time needed for real distraction. The default time is 3 seconds, but user can change it by their judgement. 

If application detects the distraction, the warning will arise. There are two ways of warning: the first is to play a beep sound when the application detects fatigue or off-job. The second way is to display a message in application interface. 

The check box below shows specific types of distraction: yawn, nodding, wink, and off-job. The user can select what the type they want to detect. An indicator shows how many times the students wink, yawn, or nod detected by the camera screen. 

The start button at the top right corner allows the user to start the detection. The stop button allows the user to stop the application. The save status button can save log of students’ detection history. Quit button means withdraw from application. The panel on the right side is a status log. It tracks and displays all the information including the time and date of the app operations. The operations include what the user input or change for settings, the camera status and the fatigue or off-job warnings.
The tools that we used for designing our porgram is Adobe XD – it was to draw the rough UI. We used Microsoft Visio to draw the diagram of flow Chart. Python was coding language to program the entire application, using Pycharm as our IDE. PyQT was GUI configuration tool. Dlib Libraries was applied to detect the actions, including fatigue, off-job features and the location of the face in the camera. We used Open CV to capture the camera. 

We met several challenges working on this project. Firstly, the algorithm design was a difficult step.  We needed to decide how to detect students’ winking, nodding, or yawning. We also had a hard time to create the UI design for application, and there were also many bugs in application that we need to fix again and again. For example, the camera could not detect wink, or it captured all expressions but couldn’t trigger warning sound. 

We solved the problems that we faced and completed our final design. In the future, we hope that our application can be integrated with machine learning to train the application to know the facial expressions of each student in a more precise way, so it can develop more accurate judgements. In addition, more user-friendly warning can be added such as changing screen colors or playing a music to warm the distracted students. Besides, the function of preventing switching screen can also be considered in this application because some students usually switch to another websites for video watching or game playing when they have online class.
