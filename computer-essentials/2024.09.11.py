'''3.4.12 문제
d m 길이의 철로 양끝에 서로 마주보고 있는 두 대의 기차 A, B가 있고, 기차 A위에는 파리 한 마리가 있다. 두 기차 A와 B가 서로를 향해 달릴 때, 파리는 기차 A에서 기차 B까지 날아가다가 기차 B와 부딪히기 직전 방향을 바꿔 기차 A에게 날아가고 다시 기차 A와 부딪히기 직전 기차 B를 향해 날아가기를 반복한다. 그러다가 기차 A, B는 부딪힌다. 이때, 기차 A의 속력은 a m/s, 기차 B의 속력은 b m/s, 파리의 속력은 c m/s 이다. d, a, b, c가 주어졌을 때, 파리가 부딪히기 직전까지 날아다닌 거리(m)를 구하는 코드를 작성하여라. 단, d, a, b, c는 모두 10000이하의 자연수이고, a, b < c < d를 만족하도록 임의로 입력한다.
'''
def fly_dist(d,a,b,c):
    return (d/(a+b))*c

d,a,b,c=map(int,input().split())
print(fly_dist(d,a,b,c))

'''3.4.13 문제
두 세 자리 수가 주어질 때, ①, ②, ③, ④에 들어갈 값을 각각 줄변경하여 출력하는 구하는 코드를 작성하여라.
'''
def print_mul(num1, num2):
    one2=num2%10
    ten2=(num2//10)%10
    hund2=num2//100
    return num1*one2, num1*ten2, num1*hund2, num1*num2

num1, num2=map(int,input().split())
print(print_mul(num1,num2), sep='\n')

'''4.2.1 문제
다음과 같은 리스트num_list가 있다.
num_list = [[1, 2, 3, [4, 5]], [6, 7, 8], 9]

    9를 츨력하는 코드를 작성하여라.

    7를 출력하는 코드를 작성하여라.

    [3, [4, 5]]를 출력하는 코드를 작성하여라.
'''

num_list = [[1, 2, 3, [4, 5]], [6, 7, 8], 9]
print(num_list[2])
print(num_list[1][1])
print(num_list[0][2:])

'''4.2.3 문제
이름과 핸드폰 번호를 input()함수로 입력받아, 아래와 같은 형식으로 출력하는 코드를 작성하여라. 이때, 사용자의 이름은 두 글자에서 네 글자까지를 입력하며, 핸드폰 번호는 아래의 형식으로 입력한다고 가정한다.
'''
namestr, phonestr=input().split(', ')

name = namestr[8:-1]
print(name)
phone = phonestr[7:-1].strip('-').strip('.')
print(phone)
# name.replace(name[1], '*', 1 # 몇개까지 바꿀 건지)
print(f"이름: {name[0]}*{name[2:]}({phone[-4:]})님 등록되었습니다.")
# print(f"이름: {name}({phone[-4:]})님 등록되었습니다.")

'''4.2.7 문제
두 문자열의 문자를 번갈아가며 결합해 하나의 문자열로 만들려고 한다.
예를 들어, dog와 cat이 주어지면, dcoagt 또는 cdaotg으로 결합한다.
두 문자열의 길이가 다르면, 가능한 데까지만 문자를 번갈아가며 결합하고 나머지는 그대로 적어준다.
예를 들어, math와 computing이 주어지면, mcaotmhputing 또는 cmoamtphuting로 결합한다.
단, 주어진 문자열들은 공백이 없다고 가정한다.

두 문자열 str1과 str2을 위의 방식으로 결합했을 때, str3가 나오면 True, 아니면 False를 출력하는 코드를 작성하자.
'''
raw1, raw2, raw3 = input().split(', ')
str1 = raw1[8:-1]
str2 = raw2[8:-1]
str3 = raw3[8:-1]

def check_str(str1, str2, str3):
    check = ''
    j = min(len(str1), len(str2))
    for i in range(j):
        check += str1[i] + str2[i]
    check += str1[j:] + str2[j:]
    if check == str3:
        return True
    else:
        return False

print(check_str(str1, str2, str3) or check_str(str2, str1, str3))

'''4.2.9 문제
두 세 자리 수가 주어질 때, ①, ②, ③, ④에 들어갈 값을 구하는 코드를 작성하여라.
(이번 장에서 배운 내용을 활용해 코드를 작성해보자.)
'''

num1 = input()
num2 = input()

def print_mul(num1, num2):
    one2 = int(num2[2])
    ten2 = int(num2[1])
    hun2 = int(num2[0])
    return int(num1)*one2, int(num1)*ten2, int(num1)*hun2, int(num1)*int(num2)
print(print_mul(num1, num2), sep='\n')


'''4.2.17 문제
(1) 문자열이 주어졌을 때, 홀수번째 문자만 추출해 출력하는 코드를 작성하라.
(2) 문자열이 주어졌을 때, 짝수번째 문자만 추출해 출력하는 코드를 작성하라.
'''

def print_odd(str):
    return str[::2]

def print_even(str):
    return str[1::2]

str = input()
print(print_odd(str))
str = input()
print(print_even(str))