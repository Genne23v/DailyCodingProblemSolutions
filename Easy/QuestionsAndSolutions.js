/*
 * Q1.
 * Given a list of numbers and a number k, return whether any two numbers from
 * the list add up to k.
 * For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
 * 17.
 * Bonus: Can you do this in one pass?
 */
function doesTwoSumHaveK(numArray, k) {
    let temp = [];
    for (let num of numArray) {
        if (temp.includes(k - num)) {
            return true;
        } else {
            temp.push(num);
        }
    }
    return false;
}

const nums = [10, 15, 3, 7];
const k = 17;
const result = doesTwoSumHaveK(nums, k);
console.log(`========= Q1 ========`);
console.log(`Do any two sums have k: ${result}\n`);

/*
 * Q2.
 * A unival tree (which stands for "universal value") is a tree where all nodes
 * under it have the same value.
 * Given the root to a binary tree, count the number of unival subtrees.
 * For example, the following tree has 5 unival subtrees:
 * "     0       "
 * "    / \      "
 * "   1   0     "
 * "      / \    "
 * "     1   0   "
 * "    / \      "
 * "   1   1     "
 */
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

class Tuple {
    constructor(isUnival, value, count) {
        this.isUnival = isUnival;
        this.value = value;
        this.count = count;
    }
}

function countUnivalSubtrees(root) {
    return helper(root).count;
}

function helper(node) {
    if (node == null) {
        return new Tuple(true, 0, 0);
    }

    const left = helper(node.left);
    const right = helper(node.right);

    let count = left.count + right.count;
    let isUnival = true;

    if (!left.isUnival || !right.isUnival) {
        isUnival = false;
    }

    if (node.left && node.left.val !== node.val) {
        isUnival = false;
    }

    if (node.right && node.right.val !== node.val) {
        isUnival = false;
    }

    if (isUnival) {
        count++;
    }

    return new Tuple(isUnival, node.val, count);
}

let root = new TreeNode(0);
root.left = new TreeNode(1);
root.right = new TreeNode(0);
root.right.left = new TreeNode(1);
root.right.right = new TreeNode(0);
root.right.left.left = new TreeNode(1);
root.right.left.right = new TreeNode(1);

const univalTreeCount = countUnivalSubtrees(root);
console.log(`========= Q2 =========`);
console.log(`Unival Tree Count: ${univalTreeCount}\n`);

/*
 * Q3.
 * Given two singly linked lists that intersect at some point, find the
 * intersecting node. The lists are non-cyclical.
 * For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return
 * the node with value 8.
 * In this example, assume nodes with the same value are the exact same node
 * objects.
 * Do this in O(M + N) time (where M and N are the lengths of the lists) and
 * constant space.
 */
class ListNode {
    constructor(val) {
        this.val = val;
        this.next = null;
    }
}

function getIntersectionNode(headA, headB) {
    const lengthA = getLength(headA);
    const lengthB = getLength(headB);

    const diff = Math.abs(lengthA - lengthB);

    let ptrA = headA;
    let ptrB = headB;

    if (lengthA > lengthB) {
        for (let i = 0; i < diff; i++) {
            ptrA = ptrA.next;
        }
    } else {
        for (let i = 0; i < diff; i++) {
            ptrB = ptrB.next;
        }
    }

    while (ptrA != null && ptrB != null) {
        if (ptrA.val === ptrB.val) {
            return ptrA;
        }
        ptrA = ptrA.next;
        ptrB = ptrB.next;
    }

    return null;
}

function getLength(node) {
    let length = 0;
    while (node != null) {
        length++;
        node = node.next;
    }
    return length;
}

let a = new ListNode(3);
a.next = new ListNode(7);
a.next.next = new ListNode(8);
a.next.next.next = new ListNode(10);

let b = new ListNode(99);
b.next = new ListNode(1);
b.next.next = new ListNode(8);
b.next.next.next = new ListNode(10);

const intersection = getIntersectionNode(a, b);
console.log('========= Q3 =========');
console.log(`Find intersecting node: ${intersection.val}\n`);

/*
 * Q4.
 * Given an array of time intervals (start, end) for classroom lectures
 * (possibly overlapping), find the minimum number of rooms required.
 * For example, given [(30, 75), (0, 50), (60, 150)], you should return 2.
 */
function minMeetingRooms(intervals) {
    if (intervals === null || intervals.length === 0) {
        return 0;
    }

    intervals.sort();
    let endTimes = [];

    endTimes.push(intervals[0][1]);

    for (let i = 1; i < intervals.length; i++) {
        const interval = intervals[i];

        /*
        If the current interval's start time is greater than or equal to the minimum end time 
        in the heap, we can reuse the same room and update the minimum end time in the heap
        */
        if (interval[0] >= endTimes[0]) {
            endTimes.pop();
        }
        endTimes.push(interval[1]);
    }

    return endTimes.length;
}

const intervals = [
    [30, 75],
    [0, 50],
    [60, 150],
];
const minimumRooms = minMeetingRooms(intervals);
console.log('========= Q4 =========');
console.log(`Minimum number of rooms: ${minimumRooms}\n`);

/*
 * Q5.
 * You are given an M by N matrix consisting of booleans that represents a
 * board. Each True boolean represents a wall. Each False boolean represents a
 * tile you can walk on.
 * Given this matrix, a start coordinate, and an end coordinate, return the
 * minimum number of steps required to reach the end coordinate from the start.
 * If there is no possible path, then return null. You can move up, left, down,
 * and right. You cannot move through walls. You cannot wrap around the edges of
 * the board.
 * For example, given the following board:
 * [[f, f, f, f],
 * [t, t, f, t],
 * [f, f, f, f],
 * [f, f, f, f]]
 * and start = (3, 0) (bottom left) and end = (0, 0) (top left), the minimum
 * number of steps required to reach the end is 7, since we would need to go
 * through (1, 2) because there is a wall everywhere else on the second row.
 */
class Cell {
    constructor(row, col) {
        this.row = row;
        this.col = col;
    }
}

class Node {
    constructor(cell, dist, parent) {
        this.cell = cell;
        this.dist = dist;
        this.parent = parent;
    }
}

function findShortestPath(board, start, end) {
    const m = board.length;
    const n = board[0].length;
    let visited = [];
    for (let i = 0; i < m; i++) {
        visited[i] = [];
        for (let j = 0; j < n; j++) {
            visited[i][j] = false;
        }
    }
    let queue = [];
    queue.push(new Node(start, 0, null));
    visited[start.row][start.col] = true;

    const directions = [
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ];
    while (queue) {
        const curr = queue.shift();
        const cell = curr.cell;
        if (cell.row === end.row && cell.col === end.col) {
            return curr.dist;
        }

        for (const dir of directions) {
            const newRow = cell.row + dir[0];
            const newCol = cell.col + dir[1];
            if (
                newRow >= 0 &&
                newRow < m &&
                newCol >= 0 &&
                newCol < n &&
                !board[newRow][newCol] &&
                !visited[newRow][newCol]
            ) {
                queue.push(
                    new Node(new Cell(newRow, newCol), curr.dist + 1, curr)
                );
                visited[newRow][newCol] = true;
            }
        }
    }

    return null;
}

const board = [
    [false, false, false, false],
    [true, true, false, true],
    [false, false, false, false],
    [false, false, false, false],
];
const start = new Cell(3, 0);
const end = new Cell(0, 0);

const steps = findShortestPath(board, start, end);
console.log(`========= Q5 =========`);
if (steps) {
    console.log(`Minimum Steps: ${steps}\n`);
} else {
    console.log('No path found\n');
}

/*
 * Q6.
 * You run an e-commerce website and want to record the last N order ids in a
 * log. Implement a data structure to accomplish this, with the following API:
 * record(order_id): adds the order_id to the log
 * get_last(i): gets the ith last element from the log. i is guaranteed to be
 * smaller than or equal to N.
 * You should be as efficient with time and space as possible.
 */
class OrderLog {
    constructor(n) {
        this.start = 0;
        this.size = 0;
        this.buffer = [];
        for (let i = 0; i < n; i++) {
            this.buffer[i] = 0;
        }
    }

    record(orderId) {
        this.buffer[(this.start + this.size) % this.buffer.length] = orderId;
        if (this.size < this.buffer.length) {
            this.size++;
        } else {
            // When the buffer is full, use start to circulate the buffer array
            this.start = (this.start + 1) % this.buffer.length;
        }
    }

    getLast(i) {
        if (i <= 0 || i > this.size) {
            throw new Error('Invalid i value');
        }
        return this.buffer[(this.start + this.size - i) % this.buffer.length];
    }
}

console.log('========= Q6 =========');
const orderLog = new OrderLog(3);
orderLog.record('1');
orderLog.record('2');
orderLog.record('3');
orderLog.record('4');
orderLog.record('5');
orderLog.record('6');
orderLog.record('7');
console.log(`Last 2th order ID: ${orderLog.getLast(2)}\n`);

/*
 * Q7.
 * Given a string of round, curly, and square open and closing brackets, return
 * whether the brackets are balanced (well-formed).
 * For example, given the string "([])[]({})", you should return true.
 * Given the string "([)]" or "((()", you should return false.
 */
function isBalanced(str) {
    let stack = [];
    for (const ch of str) {
        if (ch === '(' || ch === '{' || ch === '[') {
            stack.push(ch);
        } else if (ch === ')' || ch === '}' || ch === ']') {
            if (stack.length === 0) {
                return false;
            }
            const top = stack.pop();
            if (
                (ch === ')' && top !== '(') ||
                (ch === '}' && top !== '{') ||
                (ch === ']' && top !== '[')
            ) {
                return false;
            }
        }
    }
    return stack.length === 0;
}

const str1 = '([])[]({})';
const str2 = '([)]';
const str3 = '((()';

console.log('========= Q7 =========');
console.log(`Is str1 balanced? - ${isBalanced(str1)}`);
console.log(`Is str2 balanced? - ${isBalanced(str2)}`);
console.log(`Is str3 balanced? - ${isBalanced(str3)}\n`);

/*
 * Q8.
 * Run-length encoding is a fast and simple method of encoding strings. The
 * basic idea is to represent repeated successive characters as a single count
 * and character. For example, the string "AAAABBBCCDAA" would be encoded as
 * "4A3B2C1D2A".
 * Implement run-length encoding and decoding. You can assume the string to be
 * encoded have no digits and consists solely of alphabetic characters. You can
 * assume the string to be decoded is valid.
 */
function encode(input) {
    let encoded = '';
    let count = 1;
    let prevChar = input[0];
    for (let i = 1; i < input.length; i++) {
        const currChar = input[i];
        if (currChar === prevChar) {
            count++;
        } else {
            encoded += `${count}${prevChar}`;
            count = 1;
            prevChar = currChar;
        }
    }
    encoded += `${count}${prevChar}`;
    return encoded;
}

function decode(input) {
    let decoded = '';
    let count = 0;
    for (const ch of input) {
        if (!isNaN(parseInt(ch)) && isFinite(ch)) {
            count = count * 10 + ch;
        } else {
            for (let i = 0; i < count; i++) {
                decoded += ch;
            }
            count = 0;
        }
    }
    return decoded;
}

const input = 'AAAABBBCCDAA';
const encoded = encode(input);
const decoded = decode(encoded);

console.log('========= Q8 =========');
console.log(`Encoded string: ${encoded}`);
console.log(`Decoded string: ${decoded}\n`);

/*
 * Q9.
 * The edit distance between two strings refers to the minimum number of
 * character insertions, deletions, and substitutions required to change one
 * string to the other. For example, the edit distance between “kitten” and
 * “sitting” is three: substitute the “k” for “s”, substitute the “e” for “i”,
 * and append a “g”.
 * Given two strings, compute the edit distance between them.
 */
function editDistance(s1, s2) {
    let dp = [];
    for (let i = 0; i <= s1.length; i++) {
        dp[i] = [];
        for (let j = 0; j <= s2.length; j++) {
            dp[i][j] = 0;
        }
    }

    for (let i = 0; i <= s1.length; i++) {
        dp[i][0] = i;
    }
    for (let j = 0; j <= s2.length; j++) {
        dp[0][j] = j;
    }

    for (i = 1; i <= s1.length; i++) {
        for (j = 1; j <= s2.length; j++) {
            if (s1[i - 1] === s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // Same distance from both -1 position
            } else {
                dp[i][j] =
                    1 +
                    Math.min(
                        dp[i - 1][j - 1],
                        Math.min(dp[i][j - 1], dp[i - 1][j])
                    );
            }
        }
    }

    return dp[s1.length][s2.length];
}

const s1 = 'kitten';
const s2 = 'sitting';

/*
 * Table for kitten and sitting
 * 0 1 2 3 4 5 6 7
 * 1 1 2 3 4 5 6 7
 * 2 2 1 2 3 4 5 6
 * 3 3 2 1 2 3 4 5
 * 4 4 3 2 1 2 3 4
 * 5 5 4 3 2 2 3 4
 * 6 6 5 4 3 3 2 3
 */
console.log('========= Q9 =========');
console.log(`Edit distance of two strings: ${editDistance(s1, s2)}\n`);

/*
 * Q10.
 * Compute the running median of a sequence of numbers. That is, given a stream
 * of numbers, print out the median of the list so far on each new element.
 * Recall that the median of an even-numbered list is the average of the two
 * middle numbers.
 * For example, given the sequence [2, 1, 5, 7, 2, 0, 5], your algorithm should
 * print out:
 * 2
 * 1.5
 * 2
 * 3.5
 * 2
 * 2
 * 2
 */
class RunningMedian {
    #maxHeap = [];
    #minHeap = [];

    addNumber(num) {
        // Add num to maxHeap if num is equal or less than current heap
        if (this.#maxHeap.length === 0 || num <= this.#maxHeap[0]) {
            this.#maxHeap.push(num);
            this.#maxHeap.sort((a, b) => b - a);
        } else {
            this.#minHeap.push(num);
            this.#minHeap.sort((a, b) => a - b);
        }

        // Balance two heaps to find the middle
        if (this.#maxHeap.length - this.#minHeap.length > 1) {
            this.#minHeap.push(this.#maxHeap.shift());
            this.#minHeap.sort((a, b) => a - b);
        } else if (this.#minHeap.length - this.#maxHeap.length > 1) {
            this.#maxHeap.push(this.#minHeap.shift());
            this.#maxHeap.sort((a, b) => b - a);
        }
    }

    getMedian() {
        if (this.#maxHeap.length === 0 && this.#minHeap.length === 0) {
            throw new Error('No numbers added yet');
        }

        if (this.#maxHeap.length === this.#minHeap.length) {
            return (this.#maxHeap[0] + this.#minHeap[0]) / 2.0;
        } else if (this.#maxHeap.length > this.#minHeap.length) {
            return this.#maxHeap[0];
        } else {
            return this.#minHeap[0];
        }
    }
}

const runningMedian = new RunningMedian();
console.log('========= Q10 =========');
runningMedian.addNumber(2);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(1);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(5);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(7);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(2);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(0);
console.log(`Median: ${runningMedian.getMedian()}`);
runningMedian.addNumber(5);
console.log(`Median: ${runningMedian.getMedian()}\n`);

/*
 * Q11.
 * The power set of a set is the set of all its subsets. Write a function that,
 * given a set, generates its power set.
 * For example, given the set {1, 2, 3}, it should return {{}, {1}, {2}, {3},
 * {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}.
 * You may also use a list or array to represent a set.
 */

/*
 * Q12.
 * Implement a stack that has the following methods:
 * push(val), which pushes an element onto the stack
 * pop(), which pops off and returns the topmost element of the stack. If there
 * are no elements in the stack, then it should throw an error or return null.
 * max(), which returns the maximum value in the stack currently. If there are
 * no elements in the stack, then it should throw an error or return null.
 * Each method should run in constant time.
 */

/*
 * Q13.
 * Given a array of numbers representing the stock prices of a company in
 * chronological order, write a function that calculates the maximum profit you
 * could have made from buying and selling that stock once. You must buy before
 * you can sell it.
 * For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could
 * buy the stock at 5 dollars and sell it at 10 dollars.
 */

/*
 * Q14.
 * Suppose an arithmetic expression is given as a binary tree. Each leaf is an
 * integer and each internal node is one of '+', '−', '∗', or '/'.
 * Given the root to such a tree, write a function to evaluate it.
 * For example, given the following tree:
 * "    *      "
 * "   / \     "
 * "  +    +   "
 * " / \  / \  "
 * "3  2  4  5 "
 * You should return 45, as it is (3 + 2) * (4 + 5).
 */

/*
 * Q15.
 * Implement a URL shortener with the following methods:
 * shorten(url), which shortens the url into a six-character alphanumeric
 * string, such as zLg6wl.
 * restore(short), which expands the shortened string into the original url. If
 * no such shortened string exists, return null.
 * Hint: What if we enter the same URL twice?
 */

/*
 * Q16.
 * Given a 2D matrix of characters and a target word, write a function that
 * returns whether the word can be found in the matrix by going left-to-right,
 * or up-to-down.
 * For example, given the following matrix:
 * [['F', 'A', 'C', 'I'],
 * ['O', 'B', 'Q', 'P'],
 * ['A', 'N', 'O', 'B'],
 * ['M', 'A', 'S', 'S']]
 * and the target word 'FOAM', you should return true, since it's the leftmost
 * column. Similarly, given the target word 'MASS', you should return true,
 * since it's the last row.
 */

/*
 * Q17.
 * Given a N by M matrix of numbers, print out the matrix in a clockwise spiral.
 * For example, given the following matrix:
 * [[1, 2, 3, 4, 5],
 * [6, 7, 8, 9, 10],
 * [11, 12, 13, 14, 15],
 * [16, 17, 18, 19, 20]]
 * You should print out the following:
 * 1
 * 2
 * 3
 * 4
 * 5
 * 10
 * 15
 * 20
 * 19
 * 18
 * 17
 * 16
 * 11
 * 6
 * 7
 * 8
 * 9
 * 14
 * 13
 * 12
 */

/*
 * Q18.
 * Given a list of integers, return the largest product that can be made by
 * multiplying any three integers.
 * For example, if the list is [-10, -10, 5, 2], we should return 500, since
 * that's -10 * -10 * 5.
 * You can assume the list has at least three integers.
 */

/*
 * Q19.
 * A number is considered perfect if its digits sum up to exactly 10.
 * Given a positive integer n, return the n-th perfect number.
 * For example, given 1, you should return 19. Given 2, you should return 28.
 */

/*
 * Q20.
 * Given the head of a singly linked list, reverse it in-place.
 */
