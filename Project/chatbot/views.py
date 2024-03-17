import json
from django.shortcuts import render
from django.http import JsonResponse
from .models import PDFDocument
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .chat import get_pdf_text, get_text_chunks, get_vector_store, user_input
from .chat import user_input

@method_decorator(csrf_exempt, name='dispatch')
class chatbot(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            user_input_text = data.get('userInput', '')
            bot_responses = self.bot_response(user_input_text)
            return JsonResponse({'response': bot_responses})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    def bot_response(self, user_input_text):
        try:
            all_responses = []
            aggregated_text = ""
            for document in PDFDocument.objects.all():
                pdf_path = document.document.path
                raw_text = get_pdf_text(pdf_path)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                aggregated_text += raw_text + "\n"

            all_chunks = get_text_chunks(aggregated_text)
            get_vector_store(all_chunks)
            response = user_input(user_input_text)  # Ensure userinput returns a response
            all_responses.append(response)
        except Exception as e:
            all_responses.append({"error": str(e)})

        return all_responses


def index(request):
     return render(request,"templates/chatbot.html")

