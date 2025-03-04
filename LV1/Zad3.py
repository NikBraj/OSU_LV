numbers = []

while(1):
    try:
        num = input('Enter a number: ')
        if num == 'Done':
            break
        numbers.append(float(num))
    except:
        print('You must enter a number!')
    
print(f'Entered numbers: {len(numbers)}')
print(f'Average: {sum(numbers)/len(numbers)}')
print(f'Minimum: {min(numbers)}')
print(f'Maximun: {max(numbers)}')