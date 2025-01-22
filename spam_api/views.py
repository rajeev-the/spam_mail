from django.http import JsonResponse
from .spammodel import predictmodel

def render(request, text):

    result = predictmodel(text)
    
    
    data = {
        'is_spam': result,  
        'text': text  
    }
    
   
    return JsonResponse(data)

