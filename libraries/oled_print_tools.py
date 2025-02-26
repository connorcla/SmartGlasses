import time
from PIL import Image, ImageDraw, ImageFont
from luma.oled.device import ssd1306
from luma.core.interface.serial import spi
from luma.core.render import canvas

def print_to_screen(text, timer):

    if timer != 0 and timer < 3000:
        timer += 10
        return
    elif timer != 0:
        clear_screen(timer)
        return

    print("screen")
    serial = spi(port=0, device=0, gpio_DC=25, gpio_RST=27, gpio_CS=8)
    disp = ssd1306(serial, rotate=1)
    disp.clear()
    
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if the word can fit in the current line
        if len(current_line + " " + word) <= 9:
            current_line = current_line + " " + word if current_line else word
        else:
            # If the word can't fit, start a new line
            lines.append(current_line)
            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)
    
    with canvas(disp) as draw:
        font = ImageFont.load_default()
        x_position = 10
        for i, line in enumerate(lines):
            y_position = 10 + (i*10)
            draw.text((x_position, y_position), line, font=font, fill=255)

    timer = 10

    #time.sleep(3) #Instead of sleep, maybe have an ongoing counter involved with the state machine so we don't have to wait until the screen is cleared.
    #disp.clear()


def clear_screen(timer):
    serial = spi(port=0, device=0, gpio_DC=25, gpio_RST=27, gpio_CS=8)
    disp = ssd1306(serial, rotate=1)
    disp.clear()
    timer = 0
