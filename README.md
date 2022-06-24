# music_comprehension
music sheet reader and note to sound matcher.
 
After a long break thanks to my service at the army and the situation in the country.
I have finally returned to work on this project.
Currently I am pursuing a diffrent direction in the process of recognition of notes and general objects in the sheet
instead of removing staves and bar lines and then finding connected componnets in the image and finding what this connected component represent.
I look at the affine transform that the sol cleft has gone through from my base Sheet.
And apply the transfrom to all the objects I am searching and searching the objects with the affine transform in the Image.
In the current state I was able only to extract only the scaling transform (the sift experiment has failed my test but I will try to improve it)
also organized a little bit the project 