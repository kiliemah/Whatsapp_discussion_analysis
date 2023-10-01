from django.shortcuts import render
import os


from . import script 

def Welcome(request):
    return render(request, 'index.html')

def predictor(request):
    if request.method == "POST"  and 'discussion' in request.FILES:
        file = request.FILES['discussion']
        file_content = file.read().decode('utf-8')
        result = script.main(file_content)
        ret = result.to_numpy
    return render(request, 'result.html', {'result': ret})