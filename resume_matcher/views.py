from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from resume_matcher.utils import *
from django.conf import settings
from resume_matcher.Bert_Model.model import *
from resume_matcher.labels import *

@csrf_exempt
def matcher(request):
    print("Getting")
    if request.method == 'POST' and request.FILES['pdfFile']:
        pdf_file = request.FILES['pdfFile']
        with open(settings.MEDIA_ROOT + pdf_file.name, 'wb+') as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)
        details = handleResume(pdf_file.name)

        resume_text = ''

        for key in Keywords:
            if key in details:
                resume_text += details[key]
        
        predicted_category = predict_resume_category(resume_text)
        
        jobs = get_recommended_jobs(get_label_for_value(predicted_category.item()), resume_text)

        data = {
            'personal_details':details,
            'resume_type': get_label_for_value(predicted_category.item()),
            'recommended_jobs': jobs
        }

        return JsonResponse({'data':data})  # Send processed data back to React
    return JsonResponse({'error': 'Invalid request'})
