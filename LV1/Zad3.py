numbers = []

def selectionSort(array, size):

    for ind in range(size):
        minIndex = ind

        for num in range(ind + 1, size):

            if array[num] < array[minIndex]:
                minIndex = num
        
        (array[ind], array[minIndex]) = (array[minIndex], array[ind])


while(1):
    try:
        num = input('Enter a number: ')
        if num == 'Done':
            break
        numbers.append(float(num))
    except:
        print('You must enter a number!')

selectionSort(numbers, len(numbers))
    
print(numbers)
print(f'Entered numbers: {len(numbers)}')
print(f'Average: {sum(numbers)/len(numbers)}')
print(f'Minimum: {min(numbers)}')
print(f'Maximun: {max(numbers)}')