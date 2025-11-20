# Жадный алгоритм: интервальное планирование(7 вариант)
def interval_scheduling(intervals):
    # Сортируем интервалы по времени окончания
    intervals.sort(key=lambda x: x[1])
    result = []
    last_end = -1
    for start, end in intervals:
        # Добавляем интервал, если он не пересекается с предыдущим выбранным
        if start >= last_end:
            result.append((start, end))
            last_end = end
    return result

# Пример использования:
intervals = [(1, 3), (2, 5), (4, 7), (6, 9), (8, 10)]
selected = interval_scheduling(intervals)
print(selected)  # Выведет: [(1, 3), (4, 7), (8, 10)]
