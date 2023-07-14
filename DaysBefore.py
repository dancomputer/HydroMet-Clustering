# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:18:15 2023


y = DaysBefore(x) is a 1-D mapping from x = year relative to 1861 to y = days relative to 1861. 
i.e. x = 0 means 1861, and y=0; x =1 means 1862, and y = 365, and so on, 
accounting for leap years!


@author: danie
"""

def CheckLeap(Year):  
  # Checking if the given year is leap year  
  if((Year % 400 == 0) or  
     (Year % 100 != 0) and  
     (Year % 4 == 0)):   
    return True;
  else:
    return False;  

def DaysBefore(Year):  
  relative_to_year = 1861
  days = 0
  if Year != 0:
      for year in np.arange(relative_to_year,relative_to_year+Year):
          if CheckLeap(year) == True:
              days = days + 366
          if CheckLeap(year) == False:
              days = days + 365
  return days
