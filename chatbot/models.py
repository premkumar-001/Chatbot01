
from django.db import models

class PDFDocument(models.Model):
    document = models.FileField(upload_to='pdf_documents uploaded/')