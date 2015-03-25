Card Recognition using OpenCV
Code from the blog post 
http://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python

Aangepast door Henk-Jan van Uffelen en Maarten Kok voor de cursus Beeldherkenning te Hogeschool Utrecht.
Dit script kan een gefotografeerde speelkaart herkennen. Dit gebeurt aan de hand van trainingsplaatjes (opgeslagen in de map trainingCardFolder).
Dit zijn 52 foto's van alle verschillende kaarten. Deze zijn zo genaamd dat:

Harten aas = h1
harten 2 = h2
etc.
harten boer = h11
harten vrouw = h12
harten koning = h13

De uitkomst van het script is in ditzelfde format.


Usage:
  ./card_img.py inputCardFile trainingCardFolder
Example:
  ./card_img.py /home/student/Eindopdracht/EindopdrachtGereed/inputs/input1.jpg /home/student/Eindopdracht/EindopdrachtGereed/trainingCardFolder/