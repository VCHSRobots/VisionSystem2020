"""
fix_cocoapi.py: Script to remove troublesome line from cocoapi config file
"""

f = open("cocoapi/PythonAPI/setup.py")
lines = f.readlines()
trouble_line_ind = -1
for ind, line in enumerate(lines):
  if "extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99']" in line:
    trouble_line_ind = ind
if trouble_line_ind != -1:
  lines.pop(trouble_line_ind)
else:
  print("setup.py is safe to compile")
  exit(0)
text = ""
for line in lines:
  text += line
f.close()
f = open("cocoapi/PythonAPI/setup.py", "w")
f.write(text)
f.close()
print("line removed\nsetup.py is safe to compile")
