# Sentiment Analysis

Sentiment Analysis is seen as being to NLP what 'Hello World' is to programming languages. But, sentiment analysis is in no way just a simple one-line statement, and major companies and researches pour money, time, and other resources into improving this task. 

For this one, I've used C.J. Hutto’s [VADER](https://github.com/cjhutto/vaderSentiment) package to extract the sentiment of each book, by combining sentences together to get the overall idea.  

### First, just import everything we need 
```
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
```

### Then to read the files. I've got mine saved as 'HP1', 'HP2' etc. on the environment for easier reading. 
```
text=''
for i in range(7):
    book='HP'+str(i+1)+'.txt'
    with open(book,'r', encoding="utf8") as f:
        text += f.read()
    text += ' '
```

### NLTK tokenizes the text for you into individual sentences
```
lines = nltk.sent_tokenize(text) #tokenizes the text using fullstops
```

### Now, to get VADER
```
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vds=SentimentIntensityAnalyzer()
```

Vader displayes the negative, neutral, and positive score of the sentence. The fourth value in the dictionary, compound, gives the overall sentiment of the text, where a compound score of
* ##### >=0.05 --> Positive
* ##### <-0.05 --> Negative
* ##### <0.05 and >-0.05 --> Neutral

### Assigning each line a VADER score
````
vader_lines={}

k=0
for line in lines:
    sentiment=vds.polarity_scores(line)
    vader_lines[k]=sentiment  
    k+=1
````

### For better clarity during visualization, I have created partitions for each book to be displayed in a different colour
````
#Partitioning the lines according to the books:
book_text=''
book_split=[]
for i in range(7):
    book='HP'+str(i+1)+'.txt' #file names are HP1, HP2,...,HP7
    with open(book,'r', encoding="utf8") as f:
        book_text += f.read()
        book_lines = nltk.sent_tokenize(book_text)
        book_split.append(len(book_lines))
````        
### Since we're considering the entire series, with more than 50,000 sentences, plotting them together would not provide much insight. So, I've gathered the lines into groups of hundred, and measured the overall sentiment of each group by averaging the scores of each line in the group. 

````
x_100=[]
y_100=[]
for i in range(0,len(lines),100):
    sum=0.00
    for j in range(i,i+100):
        try:
            sum+=vader_lines[j]['compound']
        except KeyError:
            pass
    avg=sum/100
    x_100.append(i)
    y_100.append(avg)
````
### And finally, now to plot
````
k=0 #holds the line value of book_split
j=0 #index to obtain k
plt.figure(figsize = (15,10))

color=['royalblue','orange','cyan','magenta','red','brown','green']

for i in x_100:
    
    if i==x_100[-1]: #manually checking for last element because it doesn't satisfy the if condition
        print(int(k/100),':',int(i/100)) #dividing the number of lines in each book by 100 to use it with the other groups 
        a=int(k/100)
        b=int(i/100)
        plt.plot(x_100[a:b],y_100[a:b],linewidth=1.5,color=color[j])

    if i<book_split[j]:
        continue
    else:
        print(int(k/100),':',int(i/100))
        a=int(k/100)
        b=int(i/100)
        plt.plot(x_100[a:b],y_100[a:b],linewidth=1.5,color=color[j])
        k=i
        j+=1
    

plt.xticks(np.arange(0, len(lines), 5000)) 
plt.xlabel('Lines',fontsize=20)
plt.ylabel('Sentiment',fontsize=20)
plt.show() 
````
![Sentiment Analysis Screenshot](HP_Sentiment_Analysis.png?raw=true)

The most negative period in the entire series occurs between lines 48,000 to 49,000 - at the end of the sixth book. This coincides with Dumbledore's death.

There is a similar dip towards the end of the fifth book as well - Sirius's death.

One of the most positive intervals occurs between lines 22000 and 23000, in the middle of the fourth book. This coincides with Harry defeating the dragon in the Triwizard tournament, and the end of his fight with Ron (aw). 

