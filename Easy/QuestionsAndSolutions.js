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
function generatePowerSet(set) {
    let powerSet = [];
    powerSet.push([]);

    for (const element of set) {
        // Each iteration returns new Sets with element
        let newSubsets = [];
        for (const subset of powerSet) {
            let newSubset = [];
            newSubset = JSON.parse(JSON.stringify(subset));
            // Additional Set with new element
            newSubset.push(element);
            newSubsets.push(newSubset);
        }
        // console.log(powerSet, newSubsets);
        powerSet = powerSet.concat(newSubsets);
    }
    return powerSet;
}

console.log('========= Q11 =========');
const set = [1, 2, 3];
const powerSet = generatePowerSet(set);
console.log(`Power sets: `);
console.log(powerSet);
console.log(`\n`);

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
class MaxStack {
    #stack = [];
    #maxStack = [];

    push(val) {
        this.#stack.push(val);
        if (
            this.#maxStack.length === 0 ||
            val >= this.#maxStack[this.#maxStack.length - 1]
        ) {
            this.#maxStack.push(val);
        }
    }

    pop() {
        if (this.#stack.length === 0) {
            throw new Error('Stack is empty');
        }

        const val = this.#stack.pop();
        if (val === this.#maxStack[this.#maxStack.length - 1]) {
            this.#maxStack.pop();
        }
        return val;
    }

    max() {
        if (this.#maxStack.length === 0) {
            throw new Error('Stack is empty');
        }
        return this.#maxStack[this.#maxStack.length - 1];
    }
}

console.log('========= Q12 =========');
const maxStack = new MaxStack();
maxStack.push(3);
maxStack.push(1);
maxStack.push(5);
console.log(maxStack.max());
console.log(maxStack.pop());
console.log(maxStack.max());
console.log('\n');

/*
 * Q13.
 * Given a array of numbers representing the stock prices of a company in
 * chronological order, write a function that calculates the maximum profit you
 * could have made from buying and selling that stock once. You must buy before
 * you can sell it.
 * For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could
 * buy the stock at 5 dollars and sell it at 10 dollars.
 */
function maxProfit(prices) {
    if (prices === null || prices.length < 2) return 0;

    let minPrice = prices[0];
    let maxProfit = 0;
    for (let i = 1; i < prices.length; i++) {
        maxProfit = Math.max(maxProfit, prices[i] - minPrice);
        minPrice = Math.min(minPrice, prices[i]);
    }

    return maxProfit;
}

console.log('========= Q13 =========');
const stockHistory = [9, 11, 8, 5, 7, 10];
const maximumProfit = maxProfit(stockHistory);
console.log(`Max profit: ${maximumProfit}\n`);

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
function evaluate(root) {
    if (root === null) {
        return 0;
    }

    if (!root.left && !root.right) {
        return parseInt(root.val, 10);
    }

    const leftValue = evaluate(root.left);
    const rightValue = evaluate(root.right);
    switch (root.val) {
        case '+':
            return leftValue + rightValue;
        case '-':
            return leftValue - rightValue;
        case '*':
            return leftValue * rightValue;
        case '/':
            return leftValue / rightValue;
        default:
            throw new Error(`Invalid operator: ${root.val}`);
    }
}

console.log('========= Q14 =========');
const arithmeticExpTree = new TreeNode('*');
arithmeticExpTree.left = new TreeNode('+');
arithmeticExpTree.right = new TreeNode('+');
arithmeticExpTree.left.left = new TreeNode('3');
arithmeticExpTree.left.right = new TreeNode('2');
arithmeticExpTree.right.left = new TreeNode('4');
arithmeticExpTree.right.right = new TreeNode('5');
const arithmeticExpTreeResult = evaluate(arithmeticExpTree);
console.log(`Arithmetic tree evaluation: ${arithmeticExpTreeResult}\n`);

/*
 * Q15.
 * Implement a URL shortener with the following methods:
 * shorten(url), which shortens the url into a six-character alphanumeric
 * string, such as zLg6wl.
 * restore(short), which expands the shortened string into the original url. If
 * no such shortened string exists, return null.
 * Hint: What if we enter the same URL twice?
 */
class UrlShortener {
    #urlsToKeys = {};
    #keysToUrls = {};
    #BASE_URL = 'https://example.com/';

    shorten(url) {
        if (Object.keys(this.#urlsToKeys).includes(url)) {
            return this.#BASE_URL + this.#urlsToKeys[url];
        }

        const key = this.generateKey();
        this.#urlsToKeys[url] = key;
        this.#keysToUrls[key] = url;
        return this.#BASE_URL + key;
    }

    restore(shortUrl) {
        const key = shortUrl.substring(this.#BASE_URL.length);
        if (Object.keys(this.#keysToUrls).includes(key)) {
            return this.#keysToUrls[key];
        }
        return null;
    }

    generateKey() {
        let key = '';
        const characters =
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 6; i++) {
            key += characters[Math.floor(Math.random() * characters.length)];
        }
        return key;
    }
}

console.log('========= Q15 =========');
const url = 'user/create-order';
const urlShortener = new UrlShortener();
const shortenedUrl = urlShortener.shorten(url);
console.log(`Shortened URL: ${shortenedUrl}`);
const restoredUrl = urlShortener.restore(shortenedUrl);
console.log(`Restored URL: ${restoredUrl}\n`);

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
function isWordPresent(matrix, target) {
    const rows = matrix.length;
    const cols = matrix[0].length;

    for (let i = 0; i < rows; i++) {
        const rowStr = matrix[i];
        if (rowStr.includes(target)) {
            return true;
        }
    }

    for (let j = 0; j < cols; j++) {
        let colStr = '';
        for (let i = 0; i < rows; i++) {
            colStr += matrix[i][j];
        }
        if (colStr.includes(target)) {
            return true;
        }
    }
    return false;
}

console.log('========= Q16 =========');
const matrix = [
    ['F', 'A', 'C', 'I'],
    ['O', 'B', 'Q', 'P'],
    ['A', 'N', 'O', 'B'],
    ['M', 'A', 'S', 'S'],
];
const target = 'FOAM';
const isPresent = isWordPresent(matrix, target);
console.log(`Is ${target} present in the matrix: ${isPresent}\n`);

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
function printMatrixInSpiralOrder(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    let topRow = 0,
        bottomRow = rows - 1,
        leftCol = 0,
        rightCol = cols - 1;

    while (topRow <= bottomRow && leftCol <= rightCol) {
        for (let j = leftCol; j <= rightCol; j++) {
            console.log(matrix[topRow][j]);
        }
        topRow++;

        for (let i = topRow; i <= bottomRow; i++) {
            console.log(matrix[i][rightCol]);
        }
        rightCol--;

        if (topRow <= bottomRow) {
            for (let j = rightCol; j >= leftCol; j--) {
                console.log(matrix[bottomRow][j]);
            }
            bottomRow--;
        }

        if (leftCol <= rightCol) {
            for (let i = bottomRow; i >= topRow; i--) {
                console.log(matrix[i][leftCol]);
            }
            leftCol++;
        }
    }
}

console.log('========= Q17 =========');
const matrixForSpiralPrint = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
];
printMatrixInSpiralOrder(matrixForSpiralPrint);
console.log('\n');

/*
 * Q18.
 * Given a list of integers, return the largest product that can be made by
 * multiplying any three integers.
 * For example, if the list is [-10, -10, 5, 2], we should return 500, since
 * that's -10 * -10 * 5.
 * You can assume the list has at least three integers.
 */
function largestProductOfThree(nums) {
    const n = nums.length;
    let largest1 = -9999,
        largest2 = -9999,
        largest3 = -9999;
    let smallest1 = 9999,
        smallest2 = 9999;

    for (let i = 0; i < n; i++) {
        const num = nums[i];
        if (num > largest1) {
            largest3 = largest2;
            largest2 = largest1;
            largest1 = num;
        } else if (num > largest2) {
            largest3 = largest2;
            largest2 = num;
        } else if (num > largest3) {
            largest3 = num;
        }

        if (num < smallest1) {
            smallest2 = smallest1;
            smallest1 = num;
        } else if (num < smallest2) {
            smallest2 = num;
        }
    }
    return Math.max(
        largest1 * largest2 * largest3,
        largest1 * smallest1 * smallest2
    );
}

console.log('========= Q18 =========');
const listOfIntegers = [-10, -10, 5, 2];
const largestProduct = largestProductOfThree(listOfIntegers);
console.log(
    `The largest product from multiplying three numbers in the array: ${largestProduct}\n`
);

/*
 * Q19.
 * A number is considered perfect if its digits sum up to exactly 10.
 * Given a positive integer n, return the n-th perfect number.
 * For example, given 1, you should return 19. Given 2, you should return 28.
 */
function nthPerfectNumber(n) {
    let count = 0;
    let num = 19;

    while (count < n) {
        let sum = 0;
        let temp = num;

        while (temp > 0) {
            sum += temp % 10;
            temp = Math.floor(temp / 10);
        }

        if (sum === 10) {
            count++;
        }

        if (count === n) {
            return num;
        }

        num += 9;
    }

    return -1;
}

console.log('========= Q19 =========');
const n = 2;
const nthPerfectNum = nthPerfectNumber(n);
console.log(`${n}th perfect number: ${nthPerfectNum}\n`);

/*
 * Q20.
 * Given the head of a singly linked list, reverse it in-place.
 */
function reverseList(head) {
    let prev = null;
    let current = head;
    let next = null;

    while (current) {
        next = current.next;
        current.next = prev;
        prev = current;
        current = next;
    }

    return prev;
}

console.log('========= Q20 =========');
const head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);

let reversed = reverseList(head);
let output = '';
while (reversed) {
    output += `${reversed.val} -> `;
    reversed = reversed.next;
}
console.log(`${output}null\n`);

/*
 * Q21.
 * Given a list of possibly overlapping intervals, return a new list of
 * intervals where all overlapping intervals have been merged.
 * The input list is not necessarily ordered in any way.
 * For example, given [(1, 3), (5, 8), (4, 10), (20, 25)], you should return
 * [(1, 3), (4, 10), (20, 25)].
 */
class Interval {
    constructor(start, end) {
        this.start = start;
        this.end = end;
    }
}

function mergedIntervals(intervals) {
    intervals.sort((a, b) => a.start - b.start);

    let mergedIntervals = [];
    let currentInterval;

    for (const interval of intervals) {
        if (!currentInterval) {
            currentInterval = interval;
            mergedIntervals.push(currentInterval);
        } else if (interval.start <= currentInterval.end) {
            currentInterval.end = Math.max(currentInterval.end, interval.end);
        } else {
            currentInterval = interval;
            mergedIntervals.push(currentInterval);
        }
    }
    return mergedIntervals;
}

console.log('========= Q21 =========');
let listOfIntervals = [];
listOfIntervals.push(new Interval(1, 3));
listOfIntervals.push(new Interval(5, 8));
listOfIntervals.push(new Interval(4, 10));
listOfIntervals.push(new Interval(20, 25));
const intervalMerged = mergedIntervals(listOfIntervals);
console.log(`Merged intervals: ${JSON.stringify(intervalMerged)}\n`);

/*
 * Q22.
 * Given the root of a binary tree, return a deepest node. For example, in the
 * following tree, return d.
 * "    a      "
 * "   / \     "
 * "  b   c    "
 * " /         "
 * "d          "
 */
function findDeepestNode(root) {
    if (!root) {
        return;
    }

    let queue = [];
    queue.push(root);

    let deepestNode = null;

    while (queue.length > 0) {
        let size = queue.length;

        for (let i = 0; i < size; i++) {
            const node = queue.shift();

            deepestNode = node;

            if (node.left) {
                queue.push(node.left);
            }

            if (node.right) {
                queue.push(node.right);
            }
        }
    }
    return deepestNode;
}

console.log('========= Q22 =========');
const binaryRoot = new TreeNode('a');
binaryRoot.left = new TreeNode('b');
binaryRoot.right = new TreeNode('c');
binaryRoot.left.left = new TreeNode('d');

const deepestNode = findDeepestNode(binaryRoot);
console.log(`Deepest node value: ${deepestNode.val}\n`);

/*
 * Q23.
 * Given a mapping of digits to letters (as in a phone number), and a digit
 * string, return all possible letters the number could represent. You can
 * assume each valid number in the mapping is a single digit.
 * For example if {“2”: [“a”, “b”, “c”], 3: [“d”, “e”, “f”], …} then “23” should
 * return [“ad”, “ae”, “af”, “bd”, “be”, “bf”, “cd”, “ce”, “cf"].
 */
function letterCombinations(digits, digitToLetters) {
    let result = [];
    if (!digits || digits.length === 0) {
        return result;
    }

    const temp = '';
    backtrack(result, temp, digits, 0, digitToLetters);
    return result;
}

function backtrack(result, temp, digits, index, digitToLetters) {
    const copyOfTemp = temp;
    if (copyOfTemp.length === digits.length) {
        result.push(copyOfTemp);
        return;
    }

    const letters = digitToLetters[`${digits[index]}`];
    for (const letter of letters) {
        let tempPlusLetter = copyOfTemp + letter;
        backtrack(result, tempPlusLetter, digits, index + 1, digitToLetters);
        tempPlusLetter.slice(0, -1);
    }
}

console.log('========= Q23 =========');
let digitToLetters = {};
digitToLetters[2] = ['a', 'b', 'c'];
digitToLetters[3] = ['d', 'e', 'f'];
digitToLetters[4] = ['g', 'h', 'i'];
digitToLetters[5] = ['j', 'k', 'l'];
digitToLetters[6] = ['m', 'n', 'o'];
digitToLetters[7] = ['p', 'q', 'r', 's'];
digitToLetters[8] = ['t', 'u', 'v'];
digitToLetters[9] = ['w', 'x', 'y', 'z'];

const digits = '23';
lettersForDigit = letterCombinations(digits, digitToLetters);
console.log(`Letter combinations for ${digits}:`);
console.log(lettersForDigit);
console.log('\n');

/*
 * Q24.
 * Using a read7() method that returns 7 characters from a file, implement
 * readN(n) which reads n characters.
 * For example, given a file with the content “Hello world”, three read7()
 * returns “Hello w”, “orld” and then “”.
 */
// S24. This isn't tested
function read7(filename) {
    let str = '';
    let bytesRead;
    const reader = new FileReader();

    reader.addEventListener('load', (event) => {
        const fileContents = event.target.result;
        console.log(fileContents);
    });

    const file = new File([''], filename);
    const blob = file.slice(0, 7);
    reader.readAsText(blob);

    while (bytesRead !== -1) {
        str += blob;
    }
    return str;
}

function readN(n) {
    let pos = 0;
    let buffer = '';
    let str = '';

    while (str.length < n) {
        if (buffer.length === pos) {
            buffer = read7('file.txt');
            pos = 0;
            if (buffer.length === 0) break;
        }
        str += buffer[pos++];
    }
    return str;
}

console.log('========= Q24 =========');
console.log('\n');

/*
 * Q25.
 * What does the below code snippet print out? How can we fix the anonymous
 * functions to behave as we'd expect?
 *
 * functions = []
 * for i in range(10):
 * functions.append(lambda : i)
 *
 * for f in functions:
 * print(f())
 */
// It will print '9' tem times as the lambda functions created using `lambda: i`
// all reference the same variable 'i'
// CORRECTION: functions.append(lambda x=i: x)

/*
 * Q26.
 * Given a binary tree of integers, find the maximum path sum between two nodes.
 * The path must go through at least one node, and does not need to go through
 * the root.
 */
class MaxSumBinaryTree {
    #maxSum;
    constructor() {
        this.#maxSum = -9999;
    }

    maxPathSum(root) {
        this.maxSumHelper(root);
        return this.#maxSum;
    }

    maxSumHelper(node) {
        if (!node) {
            return 0;
        }

        const leftSum = Math.max(this.maxSumHelper(node.left), 0);
        const rightSum = Math.max(this.maxSumHelper(node.right), 0);

        const currentSum = node.val + Math.max(leftSum, rightSum);
        this.#maxSum = Math.max(this.#maxSum, currentSum);

        return node.val + Math.max(leftSum, rightSum);
    }
}

console.log('========= Q26 =========');
const treeForMaxSum = new TreeNode(1);
treeForMaxSum.left = new TreeNode(2);
treeForMaxSum.left.left = new TreeNode(3);
treeForMaxSum.left.right = new TreeNode(4);
treeForMaxSum.right = new TreeNode(5);
treeForMaxSum.right.left = new TreeNode(6);
treeForMaxSum.right.right = new TreeNode(7);

const maxSumBinaryTree = new MaxSumBinaryTree();
const binaryTreeMaxSum = maxSumBinaryTree.maxPathSum(treeForMaxSum);
console.log(`Binary Tree Max Sum: ${binaryTreeMaxSum}\n`);

/*
 * Q27.
 * Given a number in the form of a list of digits, return all possible
 * permutations.
 * For example, given [1,2,3], return
 * [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]].
 */
class Permutation {
    static permutations;
    constructor() {
        this.permutations = [];
    }

    permute(nums) {
        if (!nums || nums.length === 0) {
            return this.permutations;
        }

        let current = [];
        let used = [false, false, false];
        this.backtrack(nums, current, used);

        return this.permutations;
    }

    backtrack(nums, current, used) {
        if (current.length === nums.length) {
            const temp = current;
            this.permutations.push([...temp]);
            return;
        }

        for (let i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }

            current.push(nums[i]);
            used[i] = true;
            this.backtrack(nums, current, used);
            used[i] = false;
            current.pop();
        }
    }
}

console.log('========= Q27 =========');
const list = [1, 2, 3];
const permutation = new Permutation();
const permutations = permutation.permute(list);
console.log('Permutations for [1, 2, 3]: ');
console.log(permutations);
console.log('\n');

/*
 * Q28.
 * Given a 2D board of characters and a word, find if the word exists in the
 * grid.
 * The word can be constructed from letters of sequentially adjacent cell, where
 * "adjacent" cells are those horizontally or vertically neighboring. The same
 * letter cell may not be used more than once.
 * For example, given the following board:
 * [
 * ['A','B','C','E'],
 * ['S','F','C','S'],
 * ['A','D','E','E']
 * ]
 * exists(board, "ABCCED") returns true, exists(board, "SEE") returns true,
 * exists(board, "ABCB") returns false.
 */
function exists(board, word) {
    if (!board || board.length === 0 || (board[0].length === 0) | !word) {
        return false;
    }

    const m = board.length;
    const n = board[0].length;
    let visited = Array(m)
        .fill()
        .map(() => Array(n).fill(false));

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (board[i][j] === word[0]) {
                if (search(board, visited, word, i, j, 0)) {
                    return true;
                }
            }
        }
    }
    return false;
}

function search(board, visited, word, row, col, index) {
    if (index === word.length) {
        return true;
    }

    if (
        row < 0 ||
        row >= board.length ||
        col < 0 ||
        col >= board[0].length ||
        visited[row][col] ||
        board[row][col] !== word[index]
    ) {
        return false;
    }

    visited[row][col] = true;
    const found =
        search(board, visited, word, row - 1, col, index + 1) ||
        search(board, visited, word, row + 1, col, index + 1) ||
        search(board, visited, word, row, col - 1, index + 1) ||
        search(board, visited, word, row, col + 1, index + 1);
    visited[row][col] = false;

    return found;
}

console.log('========= Q28 =========');
const characterBoard = [
    ['A', 'B', 'C', 'D'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E'],
];
const word1 = 'ABCCED';
const word2 = 'SEE';
const word3 = 'ABCB';
console.log(
    `Does ${word1} exist in the board? ${exists(characterBoard, word1)}`
);
console.log(
    `Does ${word2} exist in the board? ${exists(characterBoard, word2)}`
);
console.log(
    `Does ${word3} exist in the board? ${exists(characterBoard, word3)}\n`
);

/*
 * Q29.
 * You are in an infinite 2D grid where you can move in any of the 8 directions:
 * (x,y) to
 * (x+1, y),
 * (x - 1, y),
 * (x, y+1),
 * (x, y-1),
 * (x-1, y-1),
 * (x+1,y+1),
 * (x-1,y+1),
 * (x+1,y-1)
 * You are given a sequence of points and the order in which you need to cover
 * the points. Give the minimum number of steps in which you can achieve it. You
 * start from the first point.
 * Example:
 * Input: [(0, 0), (1, 1), (1, 2)]
 * Output: 2
 * It takes 1 step to move from (0, 0) to (1, 1). It takes one more step to move
 * from (1, 1) to (1, 2).
 */
function minSteps(points) {
    let steps = 0;
    let current = points[0];
    for (let i = 1; i < points.length; i++) {
        const next = points[i];
        steps += Math.max(
            Math.abs(next[0] - current[0]),
            Math.abs(next[1] - current[1])
        );
        current = next;
    }

    return steps;
}

console.log('========= Q29 =========');
const points = [
    [0, 0],
    [1, 1],
    [1, 2],
];
const minimumSteps = minSteps(points);
console.log(`Minimum Steps: ${minimumSteps}\n`);

/*
 * Q30.
 * Given an even number (greater than 2), return two prime numbers whose sum
 * will be equal to the given number.
 * A solution will always exist. See Goldbach’s conjecture.
 * https://en.wikipedia.org/wiki/Goldbach%27s_conjecture
 * Example:
 * Input: 4
 * Output: 2 + 2 = 4
 * If there are more than one solution possible, return the lexicographically
 * smaller solution.
 * If [a, b] is one solution with a <= b, and [c, d] is another solution with c
 * <= d, then
 * [a, b] < [c, d]
 * If a < c OR a==c AND b < d.
 */
function getPrimes(n) {
    let isPrime = Array(Number(n)).fill(true);
    isPrime[0] = false;
    isPrime[1] = false;

    for (let i = 2; i * i < n; i++) {
        if (isPrime[i]) {
            for (let j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }
console.log(isPrime)
    let primes = [];
    for (let i = 2; i <= n / 2; i++) {
        if (isPrime[i] && isPrime[n - i]) {
            primes.push(i);
            primes.push(n - i);
            break;
        }
    }
    console.log(primes);
    return primes;
}

console.log('========= Q30 =========');
import { createInterface } from 'readline';

const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
});

rl.question('Enter an even number greater than 2: ', (answer) => {
    const primes = getPrimes(answer);
    console.log(`${primes[0]} + ${primes[1]} = ${answer}\n`);
    rl.close();
});

