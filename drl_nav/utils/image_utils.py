import colorsys


def rgb_to_hsv(args):
    '''convert a rgb pixel into a hsv pixel'''
    return colorsys.rgb_to_hsv(*args)

def hsv_to_rgb(args):
    '''convert a hsv pixel into a rgb pixel'''
    return colorsys.hsv_to_rgb(*args)

def is_yellow(hsv):
    '''determine is the given hsv pixel is yellow'''
    hue = hsv[0]
    sat = hsv[1]
    val = hsv[2]
    
    hue_ok = ((.105 <= hue) & (hue <= .2))
    sat_ok = .6 <= sat
    val_ok = .6 <= val
    
    return hue_ok & sat_ok & val_ok

def is_blue(hsv):
    '''determine is the given hsv pixel is blue'''
    hue = hsv[0]
    sat = hsv[1]
    val = hsv[2]
    
    hue_ok = ((.6 <= hue) & (hue <= .75))
    sat_ok = .2 <= sat
    val_ok = .3 <= val

    return hue_ok & sat_ok & val_ok