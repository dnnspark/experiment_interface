from enum import Enum

class DebugMode(Enum): 
	NONE = 0
	DEV = 1
	DEBUG = 2
	
COLORS5 = [
    # these are second-last elements of single-hue colors from http://colorbrewer2.org
    '#3182bd', # blue
    '#de2d26', # red
    '#31a354', # green
    '#756bb1', # purple 
    '#636363', # grey

]

BRIGHTER_COLORS5 = [
    # these are third elements of single-hue colors from http://colorbrewer2.org
    '#9ecae1', # blue
    '#fc9272', # red
    '#a1d99b', # green
    '#bcbddc', # purple
    '#bdbdbd', # grey
]

