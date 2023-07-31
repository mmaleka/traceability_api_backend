from django.db import models

# Create your models here.
class traceability(models.Model):
    
    serial_number = models.CharField(max_length=20, blank=True, default='')
    operator = models.CharField(max_length=20, blank=True, null=True)
    team_leader = models.CharField(max_length=20, blank=True, null=True)
    supervisor = models.CharField(max_length=20, blank=True, null=True)
    machine = models.CharField(max_length=20, blank=True, null=True)
    operation = models.CharField(max_length=20, blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)


    class Meta:
        ordering = ['created']

    def __str__(self):
        return str(self.pk) + " - " + self.serial_number


class ScannerData(models.Model):

    shell_no = models.CharField(max_length=3, blank=True, null=True)
    cast_code = models.CharField(max_length=3, blank=True, null=True)
    heat_code = models.CharField(max_length=3, blank=True, null=True) 
    shell_serial_no = models.CharField(max_length=15, blank=True, null=True)

    location = models.CharField(max_length=5, blank=True, null=True)
    added = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, blank=True, null=True)


    def __str__(self):
        return '%s %s' % (self.pk, self.shell_serial_no)

    class Meta:
        ordering = ('updated_at', )
        verbose_name = 'ScannerData'
        verbose_name_plural = 'ScannerDataPoints'