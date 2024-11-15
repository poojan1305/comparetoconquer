# Generated by Django 4.0.5 on 2022-06-25 07:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0008_city'),
    ]

    operations = [
        migrations.CreateModel(
            name='Urls',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('link', models.CharField(max_length=1000)),
                ('uuid', models.CharField(max_length=100)),
            ],
            options={
                'verbose_name': 'Urls',
                'verbose_name_plural': 'Urls',
            },
        ),
    ]