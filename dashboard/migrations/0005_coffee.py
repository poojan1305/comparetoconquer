# Generated by Django 4.0.2 on 2022-06-05 07:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0004_rename_is_finished_todo_isfinished'),
    ]

    operations = [
        migrations.CreateModel(
            name='Coffee',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=1000)),
                ('amount', models.CharField(blank=True, max_length=100)),
                ('order_id', models.CharField(max_length=1000)),
            ],
        ),
    ]