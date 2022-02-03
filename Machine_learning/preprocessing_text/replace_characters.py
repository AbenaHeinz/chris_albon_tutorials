# import library  
import re



#create text  
text_data = ["Interrobanging. By Aishwarya Henrietta", "Parking And Going By Karl Gautier", "Today is the Night By Jarek Prakash"]

#Replace Characters Method 1 
#remove periods  
remove_periods = [string.replace("."," ") for string in text_data]

#show text 
remove_periods

#Replace Characters Method 2
#create a function  
def replace_letters_with_X(string: str)->str:
    return re.sub(r'[a-zA-Z]', 'X', string)

# Apply functions
[replace_letters_with_X(string) for string in remove_periods]