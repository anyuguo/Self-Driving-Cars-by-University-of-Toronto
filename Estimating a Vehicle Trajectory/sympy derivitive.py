# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:17:09 2019

@author: AskLan
"""

import sympy as sp
from sympy import *

import math

xl,xk,yl,yk,theta,d = symbols('xl xk yl yk theta,d')

y1 = sqrt((xl - xk - d * cos(theta)) ** 2 + (yl - yk - d * sin(theta))** 2)
y2 = atan2(yl - yk - d * sin(theta), xl - xk - d * cos(theta)) - theta


py1_pxk = diff(y1, xk, 1)
py1_pyk = diff(y1, yk, 1)
py1_ptheta = diff(y1, theta, 1)


py2_pxk = diff(y2, xk, 1)
py2_pyk = diff(y2, yk, 1)
py2_ptheta = diff(y2, theta, 1)

sp.init_printing(use_latex=True)