from typing import Text
from moviepy.editor import TextClip


with open("subtitle_colors.txt", 'w') as f:
    for i in TextClip.list('color'):
        f.write(str(i)[2:-1])
        f.write('\n')

with open("subtitle_fonts.txt", 'w') as f:
    for i in TextClip.list('font'):
        f.write(str(i))
        f.write('\n')
