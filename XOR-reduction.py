def xor(a, b):
    if(a%2==0):
        rotation = [b, 1, b+1, 0]
    else:
        rotation= [a, a^b, a-1, (a-1)^b]
    return rotation[(b-a)%4]

def solution(start, length):
    checksum=0
    for i in range(0, length):
        checksum ^= xor(start+(length*i), start+(length*i)+(length-i)-1)
    return checksum

print(solution(0,3))
