la = ["1.11", "2.0.0", "1.2", "2", "0.1", "1.2.1", "1.1.1", "2.0"]

def solution(versionList):
  
  dictionary = dict.fromkeys(versionList, 0)
  for version in versionList:
    dictionary[version] = list(map(int, version.split('.')))
    
  dictionary = sorted(dictionary.items(), key=lambda x: x[1])
  return [i[0] for i in dictionary]
print(solution(la))
