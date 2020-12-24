# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:12:59 2020

@author: Akshay
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

stroke_data = pd.read_csv("E:\\SEM-2\\677\\project\\Stroke data\\STROKE.csv")

# setting the plot size for all plots
sns.set(rc={'figure.figsize':(14,8.27)})
print('Count of geneder w.r.t stroke(patinet who has stroke)')
# Consider only those who is having stroke
gender_data =stroke_data[stroke_data['stroke']==1]
# countplot for sex w.r.t. patient having stroke
sns.countplot('gender', data=gender_data)
plt.xlabel('Gender ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()
print()
print('Count of Residence type w.r.t stroke(patinet who has stroke)')
# Consider only those who is having stroke
residence_data =stroke_data[stroke_data['stroke']==1]
# countplot for residence type w.r.t. patient having stroke
sns.countplot('Residence_type', data=residence_data)
plt.xlabel('Residence type ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

print()
print('Age w.r.t Stroke(0:Patient does not have stroke 1:patient had stroke)')
#Graph of Age patiw.r.t patient having stroke or not
sns.violinplot(x="stroke",y="age", data=stroke_data);
plt.xlabel('Whether patients have  stroke or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('Age of patients', fontsize=14)
plt.show()

print()
print('Average glucose level w.r.t Stroke(0:Patient does not have stroke 1:patient had stroke)')
#gaph of avg glucose level wrt to stroke
sns.violinplot(x="stroke",y="avg_glucose_level", hue="stroke", data=stroke_data);
plt.xlabel('Whether patients have  stroke or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('Average glucose level', fontsize=14)
plt.show()


print()
print('BMI w.r.t Stroke(0:Patient does not have stroke 1:patient had stroke)')
#gaph of BMI wrt to stroke
sns.violinplot(x="stroke",y="bmi", hue="stroke", data=stroke_data);
plt.xlabel('Whether patients have  stroke or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('BMI', fontsize=14)
plt.show()



print()
print('Number for people having stroke with smoking status')
#number for people having stroke with somking status
smoker_data = stroke_data.dropna()
smoker_data  = smoker_data[smoker_data['stroke']==1]
smoker_data.smoking_status.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(10, 8));
plt.axis('equal')
plt.title('Pie Chart for smoking habbits of people');
plt.show()

print()
print('Number for people having stroke with their work type')
work_data  = stroke_data[stroke_data['stroke']==1]
work_data.work_type.value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(10, 8),colors = ['gold', 'yellowgreen', 'lightcoral', 'red']);
plt.axis('equal')
plt.title('Pie Chart for work type of people');
plt.show()


