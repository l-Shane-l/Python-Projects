def answer(pellets):
  pellets = int(pellets)
  steps = 0
  while pellets > 1:   
    if pellets & 1 == 0:
      pellets >>= 1
    else:
      pellets = (pellets - 1) if (pellets == 3 or pellets % 4 == 1) else (pellets + 1)

    steps += 1
  return steps
