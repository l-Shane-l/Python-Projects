def solution(src, dest):
    steps = 0
    movesPossible = [src]
    prevMoves = []
    while isDestPresent(movesPossible,dest) != True:
            steps = steps + 1
            movesPossible = possibleMoves(movesPossible)
            movesPossible = set(movesPossible)
            movesPossible = [x for x in movesPossible if x not in prevMoves]
            prevMoves.extend(movesPossible)
            print movesPossible
    return steps
        
    #Your code here
    
def possibleMoves(positions):
    All = []
    for i in positions:
        All.extend(GetPositions(i))
    All = [ x for x in All if -1 <= x ]
    return All

def GetPositions(pos):
    posList = []
    if pos%8 == 0 or pos == 0:
      posList = [pos+17, pos-15, pos-6, pos+10]
    elif pos%8 == 1 or pos == 1:
      posList = [pos+17, pos+10,pos+16, pos-17, pos -17, pos -6]
    elif pos%8 == 6 or pos == 6:
      posList = [pos-17, pos+15, pos+6, pos-10]
    elif pos%8 == 7 or pos == 7:
      posList = [pos-17, pos-10,pos-16, pos+17, pos +17, pos +6]   
    else:
        posList = [pos+17,pos+6,pos+15, pos-17, pos-15, pos+10,pos -10, pos-6]
    return posList

    
def isDestPresent(movesPossible,dest):
    present = False
    for i in movesPossible:
        if i == dest:
            return True
    else:
        return False


Board = []
count = 0
while count < 64:
  Board.append(count)
  count = count + 1


print (solution(0,1))
