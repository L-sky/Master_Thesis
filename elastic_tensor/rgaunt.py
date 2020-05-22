import torch

import sympy
from sympy.physics.wigner import gaunt

factor = sympy.sqrt(2)/2

def rgaunt_p(m1, m2, m3, l1, l2, l3):
    return rgaunt(l1, l2, l3, m1, m2, m3)

def rgaunt(l1, l2, l3, m1, m2, m3):
    res = None
    (m1, l1), (m2, l2), (m3, l3) = sorted([(m1, l1), (m2, l2), (m3, l3)], reverse=True)

    if m3 > 0: # then m1 > 0, m2 > 0, m3 > 0
        if m1 == m2 + m3:
            res = (-1)**m1 * factor * gaunt(l1, l2, l3, -m1, m2, m3)
        elif m2 == m1 + m3:
            res = (-1)**m2 * factor * gaunt(l1, l2, l3, m1, -m2, m3)
        elif m3 == m1 + m2:
            res = (-1)**m3 * factor * gaunt(l1, l2, l3, m1, m2, -m3)
        else:
            res = 0
    elif m3 == 0:
        if m1 == 0 and m2 == 0:
            res = gaunt(l1, l2, l3, 0, 0, 0)
        elif m1 == m2: # m1, m2 > 0
            res = (-1)**m1 * gaunt(l1, l2, l3, m1, -m1, 0)
        else:
            res = 0
    else: # m3 < 0
        if m2 < 0:
            if m1 == 0:
                if m2 == m3:
                    res = (-1)**m2 * gaunt(l1, l2, l3, 0, m2, -m2)
                else:
                    res = 0
            elif m1 > 0:
                if m2 == m1 + m3:
                    res = (-1)**m3 * factor * gaunt(l1, l2, l3, m1, -m2, m3)
                elif m3 == m1 + m2:
                    res = (-1)**m2 * factor * gaunt(l1, l2, l3, m1, m2, -m3)
                elif m1 + m2 + m3 == 0:
                    res = -(-1)**m1 * factor * gaunt(l1, l2, l3, m1, m2, m3)
                else:
                    res = 0
            else: 
                res = 0        
        else:
            res = 0
    
    return res 
