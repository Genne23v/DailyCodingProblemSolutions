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

const stringToBeEncoded = 'AAAABBBCCDAA';
const encoded = encode(stringToBeEncoded);
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
let singlyLinkedListOutput = '';
while (reversed) {
    singlyLinkedListOutput += `${reversed.val} -> `;
    reversed = reversed.next;
}
console.log(`${singlyLinkedListOutput}null\n`);

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

    let primes = [];
    for (let i = 2; i <= n / 2; i++) {
        if (isPrime[i] && isPrime[n - i]) {
            primes.push(i);
            primes.push(n - i);
            break;
        }
    }

    return primes;
}

console.log('========= Q30 =========');
const evenNum = 8;
console.log(`Two primes that sum to ${evenNum}`);
const primes = getPrimes(evenNum);
console.log(`${primes[0]} + ${primes[1]} = ${evenNum}\n`);

/*
 * Q31.
 * Determine whether a doubly linked list is a palindrome. What if it’s singly
 * linked?
 * For example, 1 -> 4 -> 3 -> 4 -> 1 returns True while 1 -> 4 returns False.
 */
class DoublyLinkedListNode {
    constructor(data) {
        this.data = data;
        this.next = null;
        this.prev = null;
    }
}

class DoublyLinkedList {
    constructor() {
        this.head = null;
        this.tail = null;
    }

    add(data) {
        let node = new DoublyLinkedListNode(data);
        if (!this.head) {
            this.head = node;
            this.tail = node;
        } else {
            this.tail.next = node;
            node.prev = this.tail;
            this.tail = node;
        }
    }

    isPalindrome() {
        let start = this.head;
        let end = this.tail;

        while (start && end) {
            if (start.data !== end.data) {
                return false;
            }
            start = start.next;
            end = end.prev;
        }
        return true;
    }
}

class SinglyLinkedListNode {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class SinglyLinkedList {
    constructor() {
        this.head = null;
    }

    add(data) {
        let node = new SinglyLinkedListNode(data);
        if (!this.head) {
            this.head = node;
        } else {
            let temp = this.head;
            while (temp.next) {
                temp = temp.next;
            }
            temp.next = node;
        }
    }

    isPalindrome() {
        if (!this.head && !this.head.next) {
            return true;
        }

        let slow = this.head;
        let fast = this.head;
        while (fast && fast.next) {
            slow = slow.next;
            fast = fast.next.next;
        }

        let secondHalf = this.reverse(slow);

        let temp1 = this.head;
        let temp2 = secondHalf;
        while (temp2) {
            if (temp1.data !== temp2.data) {
                this.reverse(secondHalf);
                return false;
            }
            temp1 = temp1.next;
            temp2 = temp2.next;
        }

        this.reverse(secondHalf);
        return true;
    }

    reverse(head) {
        let prev = null;
        let curr = head;
        let next = null;

        while (curr) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        head = prev;

        return head;
    }
}

console.log('========= Q31 =========');
const doublyLinkedList1 = new DoublyLinkedList();
doublyLinkedList1.add(1);
doublyLinkedList1.add(4);
doublyLinkedList1.add(3);
doublyLinkedList1.add(4);
doublyLinkedList1.add(1);
console.log(
    `Is the Doubly Linked List palindrome: ${doublyLinkedList1.isPalindrome()}`
);

const doublyLinkedList2 = new DoublyLinkedList();
doublyLinkedList2.add(1);
doublyLinkedList2.add(4);
console.log(
    `Is the Doubly Linked List palindrome: ${doublyLinkedList2.isPalindrome()}`
);

const singlyLinkedList1 = new SinglyLinkedList();
singlyLinkedList1.add(1);
singlyLinkedList1.add(4);
singlyLinkedList1.add(3);
singlyLinkedList1.add(4);
singlyLinkedList1.add(1);
console.log(
    `Is the Singly Linked List palindrome: ${singlyLinkedList1.isPalindrome()}`
);

const singlyLinkedList2 = new SinglyLinkedList();
singlyLinkedList2.add(1);
singlyLinkedList2.add(4);
console.log(
    `Is the Singly Linked List palindrome: ${singlyLinkedList2.isPalindrome()}`
);
console.log('\n');

/*
 * Q32.
 * Given a function f, and N return a debounced f of N milliseconds.
 * That is, as long as the debounced f continues to be invoked, f itself will
 * not be called for N milliseconds.
 */
class Debounce {
    constructor(func, delay) {
        this.delay = delay;
        this.func = func;
        this.lastExecutionTime = null;
    }

    execute() {
        const currentTime = new Date().getTime();
        if (currentTime - this.lastExecutionTime > this.delay) {
            this.lastExecutionTime = currentTime;
            this.func();
        }
    }
}

console.log('========= Q32 =========');
const debounce = new Debounce(() => console.log('Hello'), 1000);
for (let i = 0; i < 10; i++) {
    debounce.execute();
    const currentTime = new Date().getTime();
    while (new Date().getTime() - currentTime < 500) { }
}

/*
 * Q33.
 * Print the nodes in a binary tree level-wise. For example, the following
 * should print 1, 2, 3, 4, 5.
 * "   1       "
 * "  / \      "
 * " 2   3     "
 * "    / \    "
 * "   4   5   "
 */
function printLevelOrder(root) {
    if (!root) {
        return;
    }

    let queue = [];
    queue.push(root);

    while (queue.length > 0) {
        let node = queue.shift();
        console.log(node.val);

        if (node.left) {
            queue.push(node.left);
        }
        if (node.right) {
            queue.push(node.right);
        }
    }
}

console.log('========= Q33 =========');
const binaryToPrint = new TreeNode(1);
binaryToPrint.left = new TreeNode(2);
binaryToPrint.right = new TreeNode(3);
binaryToPrint.right.left = new TreeNode(4);
binaryToPrint.right.right = new TreeNode(5);

printLevelOrder(binaryToPrint);
console.log('\n');

/*
 * Q34.
 * Given two strings A and B, return whether or not A can be shifted some number
 * of times to get B.
 * For example, if A is abcde and B is cdeab, return true. If A is abc and B is
 * acb, return false.
 */
function canShift(a, b) {
    if (a.length !== b.length) {
        return false;
    }
    const a2 = a + a;
    return a2.includes(b);
}

console.log('========= Q34 =========');
const A = 'abcde';
const B = 'cdeab';
const shiftable1 = canShift(A, B);
console.log(`Can ${A} be shifted to get ${B}: ${shiftable1}`);

const C = 'abc';
const D = 'acb';
const shiftable2 = canShift(C, D);
console.log(`Can ${A} be shifted to get ${B}: ${shiftable2}`);
console.log('\n');

/*
 * Q35.
 * Given a binary tree, return the level of the tree with minimum sum.
 */
function minLevelSum(root) {
    if (!root) {
        return -1;
    }

    let queue = [];
    queue.push(root);
    let minLevel = 0;
    let minSum = Number.MAX_SAFE_INTEGER;
    let level = 0;

    while (queue.length > 0) {
        let size = queue.length;
        let sum = 0;

        for (let i = 0; i < size; i++) {
            const node = queue.shift();
            sum += node.val;

            if (node.left) {
                queue.push(node.left);
            }
            if (node.right) {
                queue.push(node.right);
            }
        }

        if (sum < minSum) {
            minSum = sum;
            minLevel = level;
        }
        level++;
    }
    return minLevel;
}

console.log('========= Q35 =========');
const minTreeSum = new TreeNode(11);
minTreeSum.left = new TreeNode(2);
minTreeSum.right = new TreeNode(3);
minTreeSum.left.left = new TreeNode(2);
minTreeSum.left.right = new TreeNode(2);
minTreeSum.right.left = new TreeNode(4);
minTreeSum.right.right = new TreeNode(5);

console.log(`Minimum Sum Tree Level: ${minLevelSum(minTreeSum)}`);
console.log('\n');

/*
 * Q36.
 * Given a sorted list of integers, square the elements and give the output in
 * sorted order.
 * For example, given [-9, -2, 0, 2, 3], return [0, 4, 4, 9, 81].
 */
function sortedSquares(nums) {
    const n = nums.length;
    let result = new Array(n);
    let left = 0;
    let right = n - 1;

    for (let i = n - 1; i >= 0; i--) {
        if (Math.abs(nums[left]) > Math.abs(nums[right])) {
            result[i] = nums[left] * nums[left];
            left++;
        } else {
            result[i] = nums[right] * nums[right];
            right--;
        }
    }

    return result;
}

console.log('========= Q36 =========');
const numArr = [-9, -2, 0, 2, 3];
const squaredArr = sortedSquares(numArr);

console.log(`Sorted Squared Array: ${squaredArr}`);
console.log('\n');

/*
 * Q37.
 * You have n fair coins and you flip them all at the same time. Any that come
 * up tails you set aside. The ones that come up heads you flip again. How many
 * rounds do you expect to play before only one coin remains?
 * Write a function that, given n, returns the number of rounds you'd expect to
 * play until one coin remains.
 */
function numRounds(n) {
    if (n === 1) {
        return 0;
    } else {
        return 1 + numRounds(Math.floor(n / 2));
    }
}

console.log('========= Q37 =========');
const numOfRounds = numRounds(10);
console.log(`Number of Rounds until one coin remaining: ${numOfRounds}`);
console.log('\n');

/*
 * Q38.
 * Given the root of a binary search tree, and a target K, return two nodes in
 * the tree whose sum equals K.
 * For example, given the following tree and K of 20
 * "    10         "
 * "   /   \       "
 * " 5      15     "
 * "       /  \    "
 * "     11    15  "
 * Return the nodes 5 and 15.
 */
function findTarget(root, k) {
    let list = [];
    inorder(root, list);

    let left = 0;
    let right = list.length - 1;
    while (left < right) {
        const sum = list[left] + list[right];
        if (sum === k) {
            let result = [];
            result.push(findNode(root, list[left]));
            result.push(findNode(root, list[right]));
            return result;
        } else if (sum < k) {
            left++;
        } else {
            right--;
        }
    }
    return null;
}

function inorder(root, list) {
    if (!root) {
        return;
    }

    inorder(root.left, list);
    list.push(root.val);
    inorder(root.right, list);
}

function findNode(root, val) {
    if (!root || root.val === val) {
        return root;
    }

    if (val < root.val) {
        return findNode(root.left, val);
    } else {
        return findNode(root.right, val);
    }
}

console.log('========= Q38 =========');
const findTwoSumBinaryTree = new TreeNode(10);
findTwoSumBinaryTree.left = new TreeNode(5);
findTwoSumBinaryTree.right = new TreeNode(15);
findTwoSumBinaryTree.right.left = new TreeNode(11);
findTwoSumBinaryTree.right.right = new TreeNode(15);

const twoNodesForK = findTarget(findTwoSumBinaryTree, 20);
console.log(`Two Nodes for K: ${twoNodesForK[0].val}, ${twoNodesForK[1].val}`);
console.log('\n');

/*
 * Q39.
 * Let's represent an integer in a linked list format by having each node
 * represent a digit in the number. The nodes make up the number in reversed
 * order.
 * For example, the following linked list:
 * 1 -> 2 -> 3 -> 4 -> 5
 * is the number 54321.
 * Given two linked lists in this format, return their sum in the same linked
 * list format.
 * For example, given
 * 9 -> 9
 * 5 -> 2
 * return 124 (99 + 25) as:
 * 4 -> 2 -> 1
 */
function addTwoNumbers(l1, l2) {
    let dummy = new ListNode(0);
    let curr = dummy;
    let carry = 0;

    while (l1 || l2) {
        let sum = carry;
        if (l1) {
            sum += l1.val;
            l1 = l1.next;
        }

        if (l2) {
            sum += l2.val;
            l2 = l2.next;
        }

        curr.next = new ListNode(sum % 10);
        curr = curr.next;
        carry = Math.floor(sum / 10);
    }

    if (carry > 0) {
        curr.next = new ListNode(carry);
    }

    return dummy.next;
}

console.log('========= Q39 =========');
const numList1 = new ListNode(9);
numList1.next = new ListNode(9);
const numList2 = new ListNode(5);
numList2.next = new ListNode(2);

let addNumList = addTwoNumbers(numList1, numList2);
while (addNumList) {
    console.log(`Add Two Numbers: ${addNumList.val}`);
    addNumList = addNumList.next;
}
console.log('\n');

/*
 * Q40.
 * Design and implement a HitCounter class that keeps track of requests (or
 * hits). It should support the following operations:
 * record(timestamp): records a hit that happened at timestamp
 * total(): returns the total number of hits recorded
 * range(lower, upper): returns the number of hits that occurred between
 * timestamps lower and upper (inclusive)
 * Follow-up: What if our system has limited memory?
 */
class HitCounter {
    constructor() {
        this.timestamps = [];
    }

    record(timestamp) {
        this.timestamps.push(timestamp);
    }

    total() {
        return this.timestamps.length;
    }

    range(lower, upper) {
        let count = 0;
        for (let i = 0; i < this.timestamps.length; i++) {
            if (this.timestamps[i] >= lower && this.timestamps[i] <= upper) {
                count++;
            }
        }
        return count;
    }
}

console.log('========= Q40 =========');
const hitCounter = new HitCounter();
hitCounter.record(Date.now());
let currentTime = Date.now();
while (Date.now() - currentTime < 100) { }
const lower = Date.now();
hitCounter.record(lower);
currentTime = Date.now();
while (Date.now() - currentTime < 100) { }
hitCounter.record(Date.now());
currentTime = Date.now();
while (Date.now() - currentTime < 100) { }
currentTime = Date.now();
const upper = currentTime;
hitCounter.record(upper);
currentTime = Date.now();
while (Date.now() - currentTime < 100) { }
hitCounter.record(Date.now());

console.log(`Total Hits: ${hitCounter.total()}`);
console.log(
    `Range Hits between ${lower} and ${upper}: ${hitCounter.range(
        lower,
        upper
    )}`
);
console.log('\n');

/*
 * Q41.
 * You have a large array with most of the elements as zero.
 * Use a more space-efficient data structure, SparseArray, that implements the
 * same interface:
 * init(arr, size): initialize with the original large array and size.
 * set(i, val): updates index at i with val.
 * get(i): gets the value at index i.
 */
class SparseArray {
    constructor(arr, size) {
        this.size = size;
        this.map = {};
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] !== 0) {
                this.map[i] = arr[i];
            }
        }
    }

    set(i, val) {
        if (i < 0 || i >= this.size) {
            throw new Error('Index out of bounds');
        }
        if (val === 0) {
            delete this.map[i];
        } else {
            this.map[i] = val;
        }
    }

    get(i) {
        if (i < 0 || i >= this.size) {
            throw new Error('Index out of bounds');
        }
        return this.map[i] || 0;
    }
}

console.log('========= Q41 =========');
const sparseArr = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1];
const sparseArray = new SparseArray(sparseArr, sparseArr.length);
sparseArray.set(0, 2);
sparseArray.set(1, 3);
console.log(`First Element in Sparse Array: ${sparseArray.get(0)}`);
console.log(`Second Element in Sparse Array: ${sparseArray.get(1)}`);
console.log('\n');

/*
 * Q42.
 * Given a binary tree, find a minimum path sum from root to a leaf.
 * For example, the minimum path in this tree is [10, 5, 1, -1], which has sum
 * 15.
 * "  10       "
 * " /  \      "
 * "5    5     "
 * " \     \   "
 * "   2    1  "
 * "       /   "
 * "     -1    "
 */
function minPathSum(root) {
    if (!root) {
        return 0;
    }
    return findMinPath(root, 0);
}

function findMinPath(node, currentSum) {
    if (!node) {
        return Number.MAX_SAFE_INTEGER;
    }
    currentSum += node.val;

    if (!node.left && !node.right) {
        return currentSum;
    }

    const leftSum = findMinPath(node.left, currentSum);
    const rightSum = findMinPath(node.right, currentSum);

    return Math.min(leftSum, rightSum);
}

console.log('========= Q42 =========');
const binaryTreeForMinSumPath = new TreeNode(10);
binaryTreeForMinSumPath.left = new TreeNode(5);
binaryTreeForMinSumPath.right = new TreeNode(5);
binaryTreeForMinSumPath.left.right = new TreeNode(2);
binaryTreeForMinSumPath.right.right = new TreeNode(1);
binaryTreeForMinSumPath.right.right.left = new TreeNode(-1);

const minSum = minPathSum(binaryTreeForMinSumPath);
console.log('Minimum path sum: ' + minSum);
console.log('\n');

/*
 * Q43.
 * Given the head of a singly linked list, swap every two nodes and return its
 * head.
 * For example, given 1 -> 2 -> 3 -> 4, return 2 -> 1 -> 4 -> 3.
 */
function swapPairs(head) {
    let dummy = new ListNode(0);
    dummy.next = head;
    let prev = dummy;

    while (head && head.next) {
        let first = head;
        let second = head.next;

        prev.next = second;
        first.next = second.next;
        second.next = first;

        prev = first;
        head = first.next;
    }

    return dummy.next;
}

console.log('========= Q43 =========');
const linkedListToSwapEveryTwo = new ListNode(1);
linkedListToSwapEveryTwo.next = new ListNode(2);
linkedListToSwapEveryTwo.next.next = new ListNode(3);
linkedListToSwapEveryTwo.next.next.next = new ListNode(4);

const swappedList = swapPairs(linkedListToSwapEveryTwo);

let current = swappedList;
while (current) {
    console.log(current.val + ' ');
    current = current.next;
}
console.log('\n');

/*
 * Q44.
 * Implement a stack API using only a heap. A stack implements the following
 * methods:
 * push(item), which adds an element to the stack
 * pop(), which removes and returns the most recently added element (or throws
 * an error if there is nothing on the stack)
 * Recall that a heap has the following operations:
 * push(item), which adds a new key to the heap
 * pop(), which removes and returns the max value of the heap
 */
class StackUsingHeap {
    constructor() {
        this.priority = 0;
        this.heap = [];
    }

    push(item) {
        this.heap.push(new HeapNode(item, this.priority++));
        this.heap.sort((a, b) => b.priority - a.priority);
    }

    pop() {
        if (this.heap.size === 0) {
            throw new Error('Stack is empty');
        }
        return this.heap.shift().value;
    }
}

class HeapNode {
    constructor(value, priority) {
        this.value = value;
        this.priority = priority;
    }
}

console.log('========= Q44 =========');
const stackUsingHeap = new StackUsingHeap();

stackUsingHeap.push(1);
stackUsingHeap.push(2);
stackUsingHeap.push(3);

console.log(stackUsingHeap.pop()); // Output: 3
console.log(stackUsingHeap.pop()); // Output: 2
console.log(stackUsingHeap.pop()); // Output: 1
console.log('\n');

/*
 * Q45.
 * Given a string, determine whether any permutation of it is a palindrome.
 * For example, carrace should return true, since it can be rearranged to form
 * racecar, which is a palindrome. daily should return false, since there's no
 * rearrangement that can form a palindrome.
 */
function isPermutationPalindrome(str) {
    const map = new Map();
    for (let i = 0; i < str.length; i++) {
        const char = str.charAt(i);
        if (map.has(char)) {
            map.delete(char);
        } else {
            map.set(char, true);
        }
    }
    return map.size <= 1;
}

console.log('========= Q45 =========');
const palindromeString = 'carrace';
const nonPalindromeString = 'daily';

console.log(
    `Is ${palindromeString} palindrome? ${isPermutationPalindrome(
        palindromeString
    )}`
); // Output: true
console.log(
    `Is ${nonPalindromeString} palindrome? ${isPermutationPalindrome(
        nonPalindromeString
    )}`
); // Output: false
console.log('\n');

/*
 * Q46.
 * Given a string, return the first recurring character in it, or null if there
 * is no recurring character.
 * For example, given the string "acbbac", return "b". Given the string
 * "abcdef", return null.
 */
function findFirstRecurringCharacter(str) {
    const map = new Map();
    for (let i = 0; i < str.length; i++) {
        const char = str.charAt(i);
        if (map.has(char)) {
            return char;
        } else {
            map.set(char, true);
        }
    }
    return null;
}

console.log('========= Q46 =========');
const strToFindRecurringString1 = 'acbbac';
const strToFindRecurringString2 = 'abcdef';

console.log(
    `First recurring char in ${strToFindRecurringString1}: ${findFirstRecurringCharacter(
        strToFindRecurringString1
    )}`
); // Output: b
console.log(
    `First recurring char in ${strToFindRecurringString2}: ${findFirstRecurringCharacter(
        strToFindRecurringString2
    )}`
); // Output: null
console.log('\n');

/*
 * Q47.
 * Given a 32-bit integer, return the number with its bits reversed.
 * For example, given the binary number 1111 0000 1111 0000 1111 0000 1111 0000,
 * return 0000 1111 0000 1111 0000 1111 0000 1111.
 */
function reverseBinary(binary) {
    let number = parseInt(binary, 2);
    let reversed = 0;

    for (let i = 0; i < 32; i++) {
        reversed = reversed << 1;
        reversed = reversed | (number & 1);
        number = number >> 1;
    }

    return reversed.toString(2);
}

console.log('========= Q47 =========');
const binary = '11110000111100001111000011110000';
console.log(reverseBinary(binary)); // Output: 00001111000011110000111100001111
console.log('\n');

/*
 * Q48.
 * You are given a list of data entries that represent entries and exits of
 * groups of people into a building. An entry looks like this:
 * {"timestamp": 1526579928, count: 3, "type": "enter"}
 * This means 3 people entered the building. An exit looks like this:
 * {"timestamp": 1526580382, count: 2, "type": "exit"}
 * This means that 2 people exited the building. timestamp is in Unix time.
 * Find the busiest period in the building, that is, the time with the most
 * people in the building. Return it as a pair of (start, end) timestamps. You
 * can assume the building always starts off and ends up empty, i.e. with 0
 * people inside.
 */
function findBusiestPeriod(data) {
    let maxCount = 0;
    let currentCount = 0;
    let maxPeriod = [0, 0];
    let maxed = false;

    for (let i = 0; i < data.length; i++) {
        const entry = data[i];
        if (entry[2] === 1) {
            currentCount += entry[1];
        } else {
            currentCount -= entry[1];
        }

        if (currentCount > maxCount) {
            maxCount = currentCount;
            maxPeriod = [entry[0], entry[0]];
            maxed = true;
        } else if (maxed && currentCount === maxCount) {
            maxPeriod[1] = entry[0];
            maxed = false;
        } else if (maxed && currentCount < maxCount) {
            maxPeriod[1] = entry[0] - 1;
            maxed = false;
        }
    }

    return maxPeriod;
}

console.log('========= Q48 =========');
const data = [
    [1526579928, 3, 1], // Entry: 3 people entered the building
    [1526579935, 2, 1], // Entry: 2 people entered the building
    [1526579940, 1, 0], // Exit: 1 person exited the building
    [1526579945, 4, 1], // Entry: 4 people entered the building
    [1526579950, 2, 0], // Exit: 2 people exited the building
    [1526579955, 1, 0], // Exit: 1 person exited the building
    [1526579960, 3, 0], // Exit: 3 people exited the building
];

const busiestPeriod = findBusiestPeriod(data);
console.log(
    'Busiest Period: [' + busiestPeriod[0] + ', ' + busiestPeriod[1] + ']'
);
console.log('\n');

/*
 * Q49.
 * Write a function to flatten a nested dictionary. Namespace the keys with a
 * period.
 * For example, given the following dictionary:
 * "{                      "
 * "    "key": 3,          "
 * "    "foo": {           "
 * "        "a": 5,        "
 * "        "bar": {       "
 * "            "baz": 8   "
 * "        }              "
 * "    }                  "
 * "}                      "
 * it should become:
 * "{                      "
 * "    "key": 3,          "
 * "    "foo.a": 5,        "
 * "    "foo.bar.baz": 8   "
 * "}                      "
 * You can assume keys do not contain dots in them, i.e. no clobbering will
 * occur.
 */
function flattenDictionary(dict) {
    const flattenedDict = {};
    flattenDictionaryHelper('', dict, flattenedDict);
    return flattenedDict;
}

function flattenDictionaryHelper(prefix, dict, flattenDic) {
    for (const key in dict) {
        if (typeof dict[key] === 'object') {
            flattenDictionaryHelper(prefix + key + '.', dict[key], flattenDic);
        } else {
            flattenDic[prefix + key] = dict[key];
        }
    }
}

console.log('========= Q49 =========');
const dict = {
    key: 3,
    foo: {
        a: 5,
        bar: {
            baz: 8,
        },
    },
};
const flattenedDict = flattenDictionary(dict);
console.log(flattenedDict);
console.log('\n');

/*
 * Q50.
 * You are given a starting state start, a list of transition probabilities for
 * a Markov chain, and a number of steps num_steps. Run the Markov chain
 * starting from start for num_steps and compute the number of times we visited
 * each state.
 * For example, given the starting state a, number of steps 5000, and the
 * following transition probabilities:
 * "[                      "
 * "  ('a', 'a', 0.9),     "
 * "  ('a', 'b', 0.075),   "
 * "  ('a', 'c', 0.025),   "
 * "  ('b', 'a', 0.15),    "
 * "  ('b', 'b', 0.8),     "
 * "  ('b', 'c', 0.05),    "
 * "  ('c', 'a', 0.25),    "
 * "  ('c', 'b', 0.25),    "
 * "  ('c', 'c', 0.5)      "
 * "]                      "
 * One instance of running this Markov chain might produce { 'a': 3012, 'b':
 * 1656, 'c': 332 }.
 */
function simulateMarkovChain(start, numSteps, transitionProbabilities) {
    const stateCount = new Map();
    stateCount.set(start, 1);

    let currentState = start;
    for (let i = 0; i < numSteps; i++) {
        const randomValue = Math.random();
        let cumulativeProbability = 0.0;

        for (const transition of transitionProbabilities) {
            if (transition.fromState === currentState) {
                cumulativeProbability += transition.probability;
                if (randomValue <= cumulativeProbability) {
                    currentState = transition.toState;
                    stateCount.set(
                        currentState,
                        stateCount.get(currentState) ?
                            stateCount.get(currentState) + 1 :
                            1
                    );
                    break;
                }
            }
        }
    }

    return stateCount;
}

class TransitionProbability {
    constructor(fromState, toState, probability) {
        this.fromState = fromState;
        this.toState = toState;
        this.probability = probability;
    }
}

console.log('========= Q50 =========');
const startState = 'a';
const numSteps = 5000;
const transitionProbabilities = [
    new TransitionProbability('a', 'a', 0.9),
    new TransitionProbability('a', 'b', 0.075),
    new TransitionProbability('a', 'c', 0.025),
    new TransitionProbability('b', 'a', 0.15),
    new TransitionProbability('b', 'b', 0.8),
    new TransitionProbability('b', 'c', 0.05),
    new TransitionProbability('c', 'a', 0.25),
    new TransitionProbability('c', 'b', 0.25),
    new TransitionProbability('c', 'c', 0.5),
];
const stateCounts = simulateMarkovChain(
    startState,
    numSteps,
    transitionProbabilities
);
console.log(`State Counts: ${JSON.stringify(Object.fromEntries(stateCounts))}`);
console.log('\n');

/*
 * Q51.
 * Determine whether there exists a one-to-one character mapping from one string
 * s1 to another s2.
 * For example, given s1 = abc and s2 = bcd, return true since we can map a to
 * b, b to c, and c to d.
 * Given s1 = foo and s2 = bar, return false since the o cannot map to two
 * characters.
 */
function isCharacterMapping(s1, s2) {
    if (s1.length !== s2.length) {
        return false;
    }

    const charMap = new Map();
    for (let i = 0; i < s1.length; i++) {
        if (charMap.has(s1[i])) {
            if (charMap.get(s1[i]) !== s2[i]) {
                return false;
            }
        } else {
            charMap.set(s1[i], s2[i]);
        }
    }

    return true;
}

console.log('========= Q51 =========');
console.log(
    `isCharacterMapping('abc', 'bcd'): ${isCharacterMapping('abc', 'bcd')}`
);
console.log(
    `isCharacterMapping('foo', 'bar'): ${isCharacterMapping('foo', 'bar')}`
);
console.log('\n');

/*
 * Q52.
 * Given a linked list and a positive integer k, rotate the list to the right by
 * k places.
 * For example, given the linked list 7 -> 7 -> 3 -> 5 and k = 2, it should
 * become 3 -> 5 -> 7 -> 7.
 * Given the linked list 1 -> 2 -> 3 -> 4 -> 5 and k = 3, it should become 3 ->
 * 4 -> 5 -> 1 -> 2.
 */
function rotateRight(head, k) {
    if (!head || k === 0) {
        return head;
    }

    let length = getLength(head);
    k = k % length;

    if (k === 0) {
        return head;
    }

    let current = head;
    for (let i = 0; i < length - k - 1; i++) {
        current = current.next;
    }

    newHead = current.next;
    current.next = null;

    let temp = newHead;
    while (temp.next) {
        temp = temp.next;
    }
    temp.next = head;

    return newHead;
}

function getLength(head) {
    let length = 0;
    let current = head;

    while (current) {
        length++;
        current = current.next;
    }
    return length;
}

function printLinkedList(head) {
    let current = head;
    let toPrint = '';
    while (current) {
        toPrint += `${current.val} -> `;
        current = current.next;
    }
    toPrint += 'null';
    console.log(toPrint);
}

console.log('========= Q52 =========');
const head1 = new ListNode(7);
head1.next = new ListNode(7);
head1.next.next = new ListNode(3);
head1.next.next.next = new ListNode(5);

const k1 = 2;
const rotated1 = rotateRight(head1, k1);
console.log(`Rotated List1: `);
printLinkedList(rotated1);

const head2 = new ListNode(1);
head2.next = new ListNode(2);
head2.next.next = new ListNode(3);
head2.next.next.next = new ListNode(4);
head2.next.next.next.next = new ListNode(5);

const k2 = 3;
const rotated2 = rotateRight(head2, k2);
console.log(`Rotated List2: `);
printLinkedList(rotated2);

console.log('\n');

/*
 * Q53.
 * Given n numbers, find the greatest common denominator between them.
 * For example, given the numbers [42, 56, 14], return 14.
 */
function gcdHelper(a, b) {
    while (b !== 0) {
        let temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

function findGreatestCommonDenominator(numbers) {
    let gcd = numbers[0];

    for (let i = 1; i < numbers.length; i++) {
        gcd = gcdHelper(gcd, numbers[i]);
    }

    return gcd;
}

console.log('========= Q53 =========');
const numbers = [42, 56, 14];
console.log(
    `Greatest Common Denominator: ${findGreatestCommonDenominator(numbers)}`
);
console.log('\n');

/*
 * Q54.
 * Given two rectangles on a 2D graph, return the area of their intersection. If
 * the rectangles don't intersect, return 0.
 * For example, given the following rectangles:
 * "{                                          "
 * "    "top_left": (1, 4),                    "
 * "    "dimensions": (3, 3) # width, height   "
 * "}                                          "
 * and
 * "{                                          "
 * "    "top_left": (0, 5),                    "
 * "    "dimensions": (4, 3) # width, height   "
 * "}                                          "
 * return 6.
 */
class Rectangle {
    constructor(topLeft, topLeftY, width, height) {
        this.topLeft = topLeft;
        this.topLeftY = topLeftY;
        this.width = width;
        this.height = height;
    }

    getIntersectionArea(other) {
        const left = Math.max(this.topLeft, other.topLeft);
        const right = Math.min(
            this.topLeft + this.width,
            other.topLeft + other.width
        );
        const top = Math.max(this.topLeftY, other.topLeftY);
        const bottom = Math.min(
            this.topLeftY + this.height,
            other.topLeftY + other.height
        );

        if (left >= right || top >= bottom) {
            return 0;
        }

        const intersectionWidth = right - left;
        const intersectionHeight = bottom - top;

        return intersectionWidth * intersectionHeight;
    }

    overlaps(other) {
        const thisRight = this.topLeft + this.width;
        const thisBottom = this.topLeftY + this.height;
        const otherRight = other.topLeft + other.width;
        const otherBottom = other.topLeftY + other.height;

        return (
            this.topLeft < otherRight &&
            thisRight > other.topLeft &&
            this.topLeftY < otherBottom &&
            thisBottom > other.topLeftY
        );
    }
}

console.log('========= Q54 =========');
const rectangle1 = new Rectangle(1, 4, 3, 3);
const rectangle2 = new Rectangle(0, 5, 4, 3);
console.log(`Intersection Area: ${rectangle1.getIntersectionArea(rectangle2)}`);
console.log('\n');

/*
 * Q55.
 * You are given given a list of rectangles represented by min and max x- and
 * y-coordinates. Compute whether or not a pair of rectangles overlap each
 * other. If one rectangle completely covers another, it is considered
 * overlapping.
 * For example, given the following rectangles:
 * "{                                          "
 * "    "top_left": (1, 4),                    "
 * "    "dimensions": (3, 3) # width, height   "
 * "},                                         "
 * "{                                          "
 * "    "top_left": (-1, 3),                   "
 * "    "dimensions": (2, 1)                   "
 * "},                                         "
 * "{                                          "
 * "    "top_left": (0, 5),                    "
 * "    "dimensions": (4, 3)                   "
 * "}                                          "
 * return true as the first and third rectangle overlap each other.
 */
function checkOverlap(rectangles) {
    for (let i = 0; i < rectangles.length; i++) {
        for (let j = i + 1; j < rectangles.length; j++) {
            if (rectangles[i].overlaps(rectangles[j])) {
                return true;
            }
        }
    }
    return false;
}

console.log('========= Q55 =========');
const rectangles = [
    new Rectangle(1, 4, 3, 3),
    new Rectangle(-1, 3, 2, 1),
    new Rectangle(0, 5, 4, 3),
];
const overlapExists = checkOverlap(rectangles);
console.log(`Overlap Exists: ${overlapExists}`);
console.log('\n');

/*
 * Q56.
 * Given an array of elements, return the length of the longest subarray where
 * all its elements are distinct.
 * For example, given the array [5, 1, 3, 5, 2, 3, 4, 1], return 5 as the
 * longest subarray of distinct elements is [5, 2, 3, 4, 1].
 */
function findLongestSubarrayLength(nums) {
    const n = nums.length;
    let maxLength = 0;
    let left = 0;
    let right = 0;
    let distinctSet = new Set();

    while (right < n) {
        if (!distinctSet.has(nums[right])) {
            distinctSet.add(nums[right]);
            maxLength = Math.max(maxLength, distinctSet.size);
            right++;
        } else {
            distinctSet.delete(nums[left]);
            left++;
        }
    }

    return maxLength;
}

console.log('========= Q56 =========');
const arrToFindLongestUniqueSubarray = [5, 1, 3, 5, 2, 3, 4, 1];
const longestSubarrayLength = findLongestSubarrayLength(
    arrToFindLongestUniqueSubarray
);
console.log(`Longest Subarray Length: ${longestSubarrayLength}`);
console.log('\n');

/*
 * Q57.
 * Given a collection of intervals, find the minimum number of intervals you
 * need to remove to make the rest of the intervals non-overlapping.
 * Intervals can "touch", such as [0, 1] and [1, 2], but they won't be
 * considered overlapping.
 * For example, given the intervals (7, 9), (2, 4), (5, 8), return 1 as the last
 * interval can be removed and the first two won't overlap.
 * The intervals are not necessarily sorted in any order.
 */
function eraseOverlapIntervals(intervals) {
    if (intervals.length === 0) {
        return 0;
    }

    intervals.sort((a, b) => a[1] - b[1]);

    let nonOverlapCount = 1;
    let end = intervals[0][1];

    for (let i = 1; i < intervals.length; i++) {
        if (intervals[i][0] >= end) {
            nonOverlapCount++;
            end = intervals[i][1];
        }
    }

    const minIntervalsToRemove = intervals.length - nonOverlapCount;

    return minIntervalsToRemove;
}

console.log('========= Q57 =========');
const collectionOfIntervals = [
    [7, 9],
    [2, 4],
    [5, 8],
];
const minIntervalsToRemove = eraseOverlapIntervals(collectionOfIntervals);
console.log(`Min Intervals To Remove: ${minIntervalsToRemove}`);
console.log('\n');

/*
 * Q58.
 * Suppose you are given two lists of n points, one list p1, p2, ..., pn on the
 * line y = 0 and the other list q1, q2, ..., qn on the line y = 1. Imagine a
 * set of n line segments connecting each point pi to qi. Write an algorithm to
 * determine how many pairs of the line segments intersect.
 */
class Segment {
    constructor(start, end) {
        this.start = start;
        this.end = end;
    }
}

function countIntersectingPairs(p, q) {
    let count = 0;
    const n = p.length;
    let segments = [];

    for (let i = 0; i < n; i++) {
        segments.push(new Segment(p[i], q[i]));
    }

    segments.sort((a, b) => a.start - b.start);

    for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < n; j++) {
            if (segments[i].end > segments[j].end) {
                count++;
            }
        }
    }

    return count;
}

console.log('========= Q58 =========');
const p = [1, 2, 3, 4];
const q = [5, 6, 7, 8];
const intersectCount = countIntersectingPairs(p, q);
console.log(`Number of Intersecting Pairs: ${intersectCount}`);
console.log('\n');

/*
 * Q59.
 * Given the root of a binary tree, find the most frequent subtree sum. The
 * subtree sum of a node is the sum of all values under a node, including the
 * node itself.
 * For example, given the following tree:
 * "  5    "
 * " / \   "
 * "2  -5  "
 * Return 2 as it occurs twice: once as the left leaf, and once as the sum of 2
 * + 5 - 5.
 */
function findMostFrequentSubtreeSum(root) {
    if (!root) {
        return 0;
    }

    let sumFrequencies = new Map();
    calculateSubtreeSum(root, sumFrequencies);

    let maxFrequency = 0;
    let mostFrequentSum = 0;

    for (const [sum, frequency] of sumFrequencies) {
        if (frequency > maxFrequency) {
            maxFrequency = frequency;
            mostFrequentSum = sum;
        }
    }

    return mostFrequentSum;
}

function calculateSubtreeSum(node, sumFrequencies) {
    if (!node) {
        return 0;
    }

    const leftSum = calculateSubtreeSum(node.left, sumFrequencies);
    const rightSum = calculateSubtreeSum(node.right, sumFrequencies);

    const currentSum = node.val + leftSum + rightSum;
    sumFrequencies.set(currentSum, sumFrequencies.get(currentSum) + 1 || 1);

    return currentSum;
}

console.log('========= Q59 =========');
const mostSubtreeSum = new TreeNode(5);
mostSubtreeSum.left = new TreeNode(2);
mostSubtreeSum.right = new TreeNode(-5);

const mostFrequentSum = findMostFrequentSubtreeSum(mostSubtreeSum);
console.log(`Most Frequent Subtree Sum: ${mostFrequentSum}`);
console.log('\n');

/*
 * Q60.
 * Given an array and a number k that's smaller than the length of the array,
 * rotate the array to the right k elements in-place.
 */
function rotate(nums, k) {
    const n = nums.length;
    k %= n;

    reverse(nums, 0, n - 1);
    // Reverse the first k elements
    reverse(nums, 0, k - 1);
    // Reverse the remaining elements
    reverse(nums, k, n - 1);
}

function reverse(nums, start, end) {
    while (start < end) {
        const temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}

console.log('========= Q60 =========');
let arrToRotate = [1, 2, 3, 4, 5];
const numToRotate = 3;

rotate(arrToRotate, numToRotate);

console.log('Rotated array: ');
let rotatedToPrint = '';
for (const num of arrToRotate) {
    rotatedToPrint += `${num} `;
}
console.log(rotatedToPrint);
console.log('\n');

/*
 * Q61.
 * You are given an array of arrays of integers, where each array corresponds to
 * a row in a triangle of numbers. For example, [[1], [2, 3], [1, 5, 1]]
 * represents the triangle:
 * "  1    "
 * " 2 3   "
 * "1 5 1  "
 * We define a path in the triangle to start at the top and go down one row at a
 * time to an adjacent value, eventually ending with an entry on the bottom row.
 * For example, 1 -> 3 -> 5. The weight of the path is the sum of the entries.
 * Write a program that returns the weight of the maximum weight path.
 */
function maximumPathSum(triangle) {
    const rows = triangle.length;

    let memo = new Array(rows);
    for (let i = 0; i < rows; i++) {
        memo[i] = new Array(rows);
    }

    // Initialize the bottom row of the memoization array with the values from the
    // triangle
    for (let i = 0; i < rows; i++) {
        memo[rows - 1][i] = triangle[rows - 1][i];
    }

    // Calculate the maximum path sum for each row from bottom to top
    for (let i = rows - 2; i >= 0; i--) {
        for (let j = 0; j <= i; j++) {
            // Compute the maximum path sum by choosing the larger adjacent value below
            memo[i][j] =
                triangle[i][j] + Math.max(memo[i + 1][j], memo[i + 1][j + 1]);
        }
    }

    return memo[0][0];
}

console.log('========= Q61 =========');
const triangle = [
    [1],
    [2, 3],
    [1, 5, 1]
];
const maxPathSum = maximumPathSum(triangle);
console.log(`Maximum Path Sum: ${maxPathSum}`);
console.log('\n');

/*
 * Q62.
 * Write a program that checks whether an integer is a palindrome. For example,
 * 121 is a palindrome, as well as 888. 678 is not a palindrome. Do not convert
 * the integer into a string.
 */
function isPalindrome(num) {
    if (num < 0 || (num !== 0 && num % 10 === 0)) {
        return false;
    }

    let reversed = 0;
    let original = num;

    while (num > 0) {
        reversed = reversed * 10 + (num % 10);
        num = Math.floor(num / 10);
    }

    return reversed === original;
}

console.log('========= Q62 =========');
const number1 = 121;
const number2 = 888;
const number3 = 678;

console.log(`${number1} is a palindrome: ${isPalindrome(number1)}`);
console.log(`${number2} is a palindrome: ${isPalindrome(number2)}`);
console.log(`${number3} is a palindrome: ${isPalindrome(number3)}`);
console.log('\n');

/*
 * Q63.
 * Given a complete binary tree, count the number of nodes in faster than O(n)
 * time. Recall that a complete binary tree has every level filled except the
 * last, and the nodes in the last level are filled starting from the left.
 */
function isCompleteBinaryTree(root) {
    const nodeCount = countNodes(root);
    const height = getHeight(root);
    const maxNodeCount = Math.pow(2, height) - 1;

    return nodeCount === maxNodeCount - 1;
}

function countNodes(node) {
    if (!node) {
        return 0;
    }

    const leftHeight = getLeftHeight(node);
    const rightHeight = getRightHeight(node);

    if (leftHeight === rightHeight) {
        return Math.pow(2, leftHeight) - 1;
    } else {
        return 1 + countNodes(node.left) + countNodes(node.right);
    }
}

function getHeight(node) {
    return getLeftHeight(node);
}

function getLeftHeight(node) {
    let height = 0;
    while (node) {
        height++;
        node = node.left;
    }
    return height;
}

function getRightHeight(node) {
    let height = 0;
    while (node) {
        height++;
        node = node.right;
    }
    return height;
}

console.log('========= Q63 =========');
const completeBinaryTree = new TreeNode(1);
completeBinaryTree.left = new TreeNode(2);
completeBinaryTree.right = new TreeNode(3);
completeBinaryTree.left.left = new TreeNode(4);
completeBinaryTree.left.right = new TreeNode(5);
completeBinaryTree.right.left = new TreeNode(6);

console.log(
    `Is this binary tree complete? (true/false): ${isCompleteBinaryTree(
        completeBinaryTree
    )}`
);

/*
 * Q64.
 * Given an integer, find the next permutation of it in absolute order. For
 * example, given 48975, the next permutation would be 49578.
 */
function findNextPermutation(num) {
    const digits = num.toString().split('');

    // Find the first decreasing digit from right to left
    const pivotIndex = findPivotIndex(digits);

    // If no pivot is found, return the integer itself
    if (pivotIndex === -1) {
        return num;
    }

    // Find the smallest digit greater than pivot to the right of pivot
    const swapIndex = findSwapIndex(digits, pivotIndex);

    // Swap pivot and swap
    swap(digits, pivotIndex, swapIndex);

    // Reverse the digits to the right of pivot
    reverseNumber(digits, pivotIndex + 1, digits.length - 1);

    return convertToInt(digits);
}

function findPivotIndex(digits) {
    for (let i = digits.length - 2; i >= 0; i--) {
        if (digits[i] < digits[i + 1]) {
            return i;
        }
    }
    return -1;
}

function findSwapIndex(digits, pivotIndex) {
    for (let i = digits.length - 1; i > pivotIndex; i--) {
        if (digits[i] > digits[pivotIndex]) {
            return i;
        }
    }
}

function swap(digits, i, j) {
    const temp = digits[i];
    digits[i] = digits[j];
    digits[j] = temp;
}

function reverseNumber(digits, start, end) {
    while (start < end) {
        swap(digits, start, end);
        start++;
        end--;
    }
}

function convertToInt(digits) {
    return parseInt(digits.join(''));
}

console.log('========= Q64 =========');
const numToFindNextPermutation = 48975;
const nextPermutation = findNextPermutation(numToFindNextPermutation);
console.log(
    `Next permutation of ${numToFindNextPermutation}: ${nextPermutation}`
);
console.log('\n');

/*
 * Q65.
 * A permutation can be specified by an array P, where P[i] represents the
 * location of the element at i in the permutation. For example, [2, 1, 0]
 * represents the permutation where elements at the index 0 and 2 are swapped.
 * Given an array and a permutation, apply the permutation to the array. For
 * example, given the array ["a", "b", "c"] and the permutation [2, 1, 0],
 * return ["c", "b", "a"].
 */
function applyPermutation(array, permutation) {
    if (array.length !== permutation.length) {
        throw new Error('Array and permutation must be of the same length');
    }

    let result = new Array(array.length);

    for (let i = 0; i < array.length; i++) {
        result[permutation[i]] = array[i];
    }

    return result;
}

console.log('========= Q65 =========');
const arrayToPermute = ['a', 'b', 'c'];
const permutationRule = [2, 1, 0];
const resultFromPermutation = applyPermutation(arrayToPermute, permutationRule);
console.log(`Permutation result: ${resultFromPermutation}`);
console.log('\n');

/*
 * Q66.
 * A Collatz sequence in mathematics can be defined as follows. Starting with
 * any positive integer:
 * if n is even, the next number in the sequence is n / 2
 * if n is odd, the next number in the sequence is 3n + 1
 * It is conjectured that every such sequence eventually reaches the number 1.
 * Test this conjecture.
 * Bonus: What input n <= 1000000 gives the longest sequence?
 */
function collatzSequence(n) {
    if (n <= 0) {
        throw new Error('n must be a positive integer');
    }

    if (n === 1) {
        return 0;
    }

    if (n % 2 === 0) {
        return 1 + collatzSequence(n / 2);
    } else {
        return 1 + collatzSequence(3 * n + 1);
    }
}

let longestSequence = 0;
let longestSequenceNumber = 0;

for (let i = 1; i <= 1000000; i++) {
    const sequenceLength = collatzSequence(i);
    if (sequenceLength > longestSequence) {
        longestSequence = sequenceLength;
        longestSequenceNumber = i;
    }
}

console.log('========= Q66 =========');
console.log(`Longest sequence: ${longestSequence}`);
console.log(`Input n: ${longestSequenceNumber}`);
console.log('\n');

/*
 * Q67.
 * Spreadsheets often use this alphabetical encoding for its columns: "A", "B",
 * "C", ..., "AA", "AB", ..., "ZZ", "AAA", "AAB", ....
 * Given a column number, return its alphabetical column id. For example, given
 * 1, return "A". Given 27, return "AA".
 */
function getColumnID(columnNumber) {
    let columnID = '';

    while (columnNumber > 0) {
        const remainder = (columnNumber - 1) % 26;
        const ch = String.fromCharCode(65 + remainder);
        columnID += ch;
        columnNumber = Math.floor(columnNumber / 26);
    }

    return columnID;
}

console.log('========= Q67 =========');
const columnNumber1 = 1;
const columnNumber2 = 27;

const columnID1 = getColumnID(columnNumber1);
const columnID2 = getColumnID(columnNumber2);

console.log(`Column ID for ${columnNumber1}: ${columnID1}`);
console.log(`Column ID for ${columnNumber2}: ${columnID2}`);
console.log('\n');

/*
 * Q68.
 * Given an integer n, return the length of the longest consecutive run of 1s in
 * its binary representation.
 * For example, given 156, you should return 3.
 */
function longestConsecutiveRun(n) {
    const binary = n.toString(2);
    let maxLength = 0;
    let currentLength = 0;

    for (let i = 0; i < binary.length; i++) {
        if (binary[i] === '1') {
            currentLength++;
        } else {
            maxLength = Math.max(maxLength, currentLength);
            currentLength = 0;
        }
    }

    return maxLength;
}

console.log('========= Q68 =========');
const numToFindLongest1s = 156;
const longestRun = longestConsecutiveRun(numToFindLongest1s);
console.log(`Longest run of 1s in ${numToFindLongest1s}: ${longestRun}`);
console.log('\n');

/*
 * Q69.
 * Let's define a "sevenish" number to be one which is either a power of 7, or
 * the sum of unique powers of 7. The first few sevenish numbers are 1, 7, 8,
 * 49, and so on. Create an algorithm to find the nth sevenish number.
 */
function getNthSevenishNumber(n) {
    let sevenishNumbers = new Array(n);
    sevenishNumbers[0] = 1;

    let nextPowerOf7Index = 1;
    let powerIndex = 1;
    let nextIndexForSum = 1;
    let currentPowerOf7 = 7;
    let nextPowerOf7 = 7;

    for (let i = 1; i < n; i++) {
        if (i === nextPowerOf7Index) {
            sevenishNumbers[i] = nextPowerOf7;

            nextPowerOf7Index += Math.pow(2, powerIndex);
            powerIndex++;

            currentPowerOf7 = nextPowerOf7;
            nextPowerOf7 *= 7;
            nextIndexForSum = 0;
        } else {
            sevenishNumbers[i] =
                sevenishNumbers[nextIndexForSum] + currentPowerOf7;
            nextIndexForSum++;
        }
    }

    return sevenishNumbers[n - 1];
}

console.log('========= Q69 =========');
const nthSevenish = 5;
const nthSevenishNumber = getNthSevenishNumber(nthSevenish);
console.log(`The ${nthSevenish}th sevenish number: ${nthSevenishNumber}`);
console.log('\n');

/*
 * Q70.
 * Given a sorted array, find the smallest positive integer that is not the sum
 * of a subset of the array.
 * For example, for the input [1, 2, 3, 10], you should return 7.
 * Do this in O(N) time.
 */
function findSmallestPositiveInteger(nums) {
    let smallestPositiveInteger = 1;

    for (let i = 0; i < nums.length; i++) {
        if (nums[i] > smallestPositiveInteger) {
            break;
        }
        smallestPositiveInteger += nums[i];
    }

    return smallestPositiveInteger;
}

console.log('========= Q70 =========');
const numsToFindSmallestInteger = [1, 2, 3, 10];
const smallestInteger = findSmallestPositiveInteger(numsToFindSmallestInteger);
console.log(`Smallest positive integer: ${smallestInteger}`);
console.log('\n');

/*
 * Q71.
 * There are N prisoners standing in a circle, waiting to be executed. The
 * executions are carried out starting with the kth person, and removing every
 * successive kth person going clockwise until there is no one left.
 * Given N and k, write an algorithm to determine where a prisoner should stand
 * in order to be the last survivor.
 * For example, if N = 5 and k = 2, the order of executions would be [2, 4, 1,
 * 5, 3], so you should return 3.
 * Bonus: Find an O(log N) solution if k = 2.
 */
function findLastPrisoner(N, k) {
    if (k === 2) {
        // O(log N) solution for k = 2
        const highestPowerOfTwo = highestPowerOf2(N);
        return 2 * (N - highestPowerOfTwo) + 1;
    }

    let prisoners = [];
    for (let i = 1; i <= N; i++) {
        prisoners.push(i);
    }

    let index = 0;
    while (prisoners.length > 1) {
        // Compute the next index to remove
        index = (index + k - 1) % prisoners.length;

        // Remove the prisoner at the computed index
        prisoners.splice(index, 1);
    }

    return prisoners[0];
}

function highestPowerOf2(N) {
    let powerOf2 = 1;
    while (powerOf2 * 2 <= N) {
        powerOf2 *= 2;
    }
    return powerOf2;
}

console.log('========= Q71 =========');
const nPrisoners = 5;
const kthPrisoner = 2;
const lastSurvivor = findLastPrisoner(nPrisoners, kthPrisoner);
console.log(`The last survivor's position is: ${lastSurvivor}`);
console.log('\n');

/*
 * Q72.
 * Boggle is a game played on a 4 x 4 grid of letters. The goal is to find as
 * many words as possible that can be formed by a sequence of adjacent letters
 * in the grid, using each cell at most once. Given a game board and a
 * dictionary of valid words, implement a Boggle solver.
 */
class TrieNode {
    static #ALPHABET_SIZE = 26;

    constructor() {
        this.children = Array(TrieNode.#ALPHABET_SIZE).fill(null);
        this.isEndOfWord = false;
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }

    insert(word) {
        let current = this.root;

        for (let i = 0; i < word.length; i++) {
            const index = word.charCodeAt(i) - 'A'.charCodeAt(0);

            if (!current.children[index]) {
                current.children[index] = new TrieNode();
            }
            current = current.children[index];
        }

        current.isEndOfWord = true;
    }

    search(word) {
        let current = this.root;

        for (let i = 0; i < word.length; i++) {
            const index = word.charCodeAt(i) - 'A'.charCodeAt(0);

            if (!current.children[index]) {
                return false;
            }
            current = current.children[index];
        }
        return current.isEndOfWord;
    }
}

function findWords(board, dictionary) {
    let trie = new Trie();
    for (const word of dictionary) {
        trie.insert(word);
    }

    let foundWords = new Set();
    const rows = board.length;
    const columns = board[0].length;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            let visited = new Array(rows)
                .fill(false)
                .map(() => new Array(columns).fill(false));
            let currentWord = '';
            dfs(board, visited, i, j, currentWord, trie, foundWords);
        }
    }

    return foundWords;
}

function dfs(board, visited, row, col, currentWord, trie, foundWords) {
    const rows = board.length;
    const columns = board[0].length;

    if (
        row < 0 ||
        row >= rows ||
        col < 0 ||
        col >= columns ||
        visited[row][col]
    ) {
        return;
    }

    currentWord += board[row][col];

    if (trie.search(currentWord)) {
        foundWords.add(currentWord);
    }

    visited[row][col] = true;
    dfs(board, visited, row - 1, col, currentWord, trie, foundWords);
    dfs(board, visited, row + 1, col, currentWord, trie, foundWords);
    dfs(board, visited, row, col - 1, currentWord, trie, foundWords);
    dfs(board, visited, row, col + 1, currentWord, trie, foundWords);
    dfs(board, visited, row - 1, col - 1, currentWord, trie, foundWords);
    dfs(board, visited, row - 1, col + 1, currentWord, trie, foundWords);
    dfs(board, visited, row + 1, col - 1, currentWord, trie, foundWords);
    dfs(board, visited, row + 1, col + 1, currentWord, trie, foundWords);

    currentWord = currentWord.substring(0, currentWord.length - 1);
    visited[row][col] = false;
}

console.log('========= Q72 =========');
const boggleBoard = [
    ['A', 'B', 'C', 'D'],
    ['E', 'F', 'G', 'H'],
    ['I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P'],
];
const dictionary = ['ABEF', 'BFJ', 'CDEG', 'PONM', 'PONMLKJIHGFEDCBA'];
let foundWords = findWords(boggleBoard, dictionary);
console.log(`Found words: ${JSON.stringify(Array.from(foundWords))}`);
console.log('\n');

/*
 * Q73.
 * Given a string with repeated characters, rearrange the string so that no two
 * adjacent characters are the same. If this is not possible, return None.
 * For example, given "aaabbc", you could return "ababac". Given "aaab", return
 * None.
 */
function rearrangeString(input) {
    let charFrequency = new Map();
    for (const char of input) {
        charFrequency.set(char, charFrequency.get(char) + 1 || 1);
    }

    let priorityQueue = Array.from(charFrequency.keys());
    priorityQueue.sort();

    let rearrangedString = '';
    while (priorityQueue.length > 0) {
        const currentChar = priorityQueue.shift();

        if (
            rearrangedString.length >= 1 &&
            rearrangedString.charAt(rearrangedString.length - 1) === currentChar
        ) {
            if (priorityQueue.length === 0) {
                return null;
            }

            const nextChar = priorityQueue.shift();
            rearrangedString += nextChar;

            let frequency = charFrequency.get(nextChar);
            frequency--;
            if (frequency > 0) {
                charFrequency.set(nextChar, frequency);
                priorityQueue.push(nextChar);
            }

            priorityQueue.push(currentChar);
        } else {
            rearrangedString += currentChar;

            let frequency = charFrequency.get(currentChar);
            frequency--;
            if (frequency > 0) {
                charFrequency.set(currentChar, frequency);
                priorityQueue.push(currentChar);
            }
        }
        priorityQueue.sort();
    }

    return rearrangedString;
}

console.log('========= Q73 =========');
const input1 = 'aaabbc';
const rearranged1 = rearrangeString(input1);
console.log(`Rearranged string for ${input1}: ${rearranged1}`);

const input2 = 'aaab';
const rearranged2 = rearrangeString(input2);
console.log(`Rearranged string for ${input2}: ${rearranged2}`);
console.log('\n');

/*
 * Q74.
 * Implement a PrefixMapSum class with the following methods:
 * insert(key: str, value: int): Set a given key's value in the map. If the key
 * already exists, overwrite the value.
 * sum(prefix: str): Return the sum of all values of keys that begin with a
 * given prefix.
 * For example, you should be able to run the following code:
 * mapsum.insert("columnar", 3)
 * assert mapsum.sum("col") == 3
 *
 * mapsum.insert("column", 2)
 * assert mapsum.sum("col") == 5
 */
class PrefixMapSum {
    constructor() {
        this.map = new Map();
    }

    insert(key, value) {
        this.map.set(key, value);
    }

    sum(prefix) {
        let sum = 0;
        for (const [key, value] of this.map) {
            if (key.startsWith(prefix)) {
                sum += value;
            }
        }
        return sum;
    }
}

console.log('========= Q74 =========');
let mapsum = new PrefixMapSum();
mapsum.insert('columnar', 3);
console.log(`Sum of 'col': ${mapsum.sum('col')}`);

mapsum.insert('column', 2);
console.log(`Sum of 'col': ${mapsum.sum('col')}`);
console.log('\n');

/*
 * Q75.
 * Implement the function fib(n), which returns the nth number in the Fibonacci
 * sequence, using only O(1) space.
 */
function fib(n) {
    if (n <= 1) {
        return n;
    }

    let prevPrev = 0;
    let prev = 1;
    let current = 0;

    for (let i = 2; i <= n; i++) {
        current = prev + prevPrev;
        prevPrev = prev;
        prev = current;
    }

    return current;
}

console.log('========= Q75 =========');
const nthFib = 6;
console.log(`Fibonacci of ${nthFib}: ${fib(nthFib)}`);
console.log('\n');

/*
 * Q76.
 * A tree is symmetric if its data and shape remain unchanged when it is
 * reflected about the root node. The following tree is an example:
 * "        4          "
 * "      / | \        "
 * "    3   5   3      "
 * "  /           \    "
 * "9              9   "
 * Given a k-ary tree, determine whether it is symmetric.
 */
class NonBinaryTreeNode {
    constructor(value) {
        this.value = value;
        this.children = [];
    }
}

function isSymmetric(root) {
    if (!root) {
        return true;
    }

    return isMirror(root, root);
}

function isMirror(node1, node2) {
    if (!node1 && !node2) {
        return true;
    }

    if (!node1 || !node2) {
        return false;
    }

    if (node1.value !== node2.value) {
        return false;
    }

    if (node1.children.length !== node2.children.length) {
        return false;
    }

    const size = node1.children.length;
    for (let i = 0; i < size; i++) {
        if (!isMirror(node1.children[i], node2.children[size - 1 - i])) {
            return false;
        }
    }

    return true;
}

console.log('========= Q76 =========');
const symmetricTree = new NonBinaryTreeNode(4);
const node1 = new NonBinaryTreeNode(3);
const node2 = new NonBinaryTreeNode(5);
const node3 = new NonBinaryTreeNode(3);
const node4 = new NonBinaryTreeNode(9);
const node5 = new NonBinaryTreeNode(9);

symmetricTree.children.push(node1);
symmetricTree.children.push(node2);
symmetricTree.children.push(node3);
node1.children.push(node4);
node3.children.push(node5);

console.log(`Is the tree symmetric? ${isSymmetric(symmetricTree)}`);
console.log('\n');

/*
 * Q77.
 * In academia, the h-index is a metric used to calculate the impact of a
 * researcher's papers. It is calculated as follows:
 * A researcher has index h if at least h of her N papers have h citations each.
 * If there are multiple h satisfying this formula, the maximum is chosen.
 * For example, suppose N = 5, and the respective citations of each paper are
 * [4, 3, 0, 1, 5]. Then the h-index would be 3, since the researcher has 3
 * papers with at least 3 citations.
 * Given a list of paper citations of a researcher, calculate their h-index.
 */
function calculateHIndex(citations) {
    citations.sort((a, b) => a - b);

    let hIndex = 0;
    const n = citations.length;
    for (let i = 0; i < n; i++) {
        const smallerCount = Math.min(citations[i], n - i);
        hIndex = Math.max(hIndex, smallerCount);
    }

    return hIndex;
}

console.log('========= Q77 =========');
const citations = [4, 3, 0, 1, 5];
console.log(`H-index of ${citations}: ${calculateHIndex(citations)}`);
console.log('\n');

/*
 * Q78.
 * The Sieve of Eratosthenes is an algorithm used to generate all prime numbers
 * smaller than N. The method is to take increasingly larger prime numbers, and
 * mark their multiples as composite.
 * For example, to find all primes less than 100, we would first mark [4, 6, 8,
 * ...] (multiples of two), then [6, 9, 12, ...] (multiples of three), and so
 * on. Once we have done this for all primes less than N, the unmarked numbers
 * that remain will be prime.
 * Implement this algorithm.
 * Bonus: Create a generator that produces primes indefinitely (that is, without
 * taking N as an input).
 */
function generatePrimes(n) {
    let isComposite = new Array(n + 1).fill(false);
    let primes = [];

    for (let i = 2; i <= n; i++) {
        if (!isComposite[i]) {
            primes.push(i);

            for (let j = i * i; j <= n; j += i) {
                isComposite[j] = true;
            }
        }
    }

    return primes;
}

class PrimeGenerator {
    #primes;

    constructor() {
        this.#primes = [];
        this.#primes.push(2);
    }

    generate() {
        let n = this.#primes[this.#primes.length - 1] + 1;
        while (true) {
            if (this.isPrime(n)) {
                this.#primes.push(n);
                return n;
            }
            n++;
        }
    }

    isPrime(n) {
        for (let i = 0; i < this.#primes.length; i++) {
            if (n % this.#primes[i] === 0) {
                return false;
            }
        }
        return true;
    }
}

console.log('========= Q78 =========');
const nPrime = 100;
const primeNums = generatePrimes(nPrime);
console.log(`Prime numbers less than ${nPrime}: ${primeNums}`);

const primeGenerator = new PrimeGenerator();
const nForPrimeGenerator = 20;
console.log(`First ${nForPrimeGenerator} prime numbers:`);

let primeNumString = '';
for (let i = 0; i < nForPrimeGenerator; i++) {
    primeNumString += `${primeGenerator.generate()}, `;
}
console.log(primeNumString.slice(0, -2));
console.log('\n');

/*
 * Q79.
 * Given a binary tree, determine whether or not it is height-balanced. A
 * height-balanced binary tree can be defined as one in which the heights of the
 * two subtrees of any node never differ by more than one.
 */
class BalancedBinaryTree {
    isBalanced(root) {
        if (!root) {
            return true;
        }

        const leftHeight = this.getHeight(root.left);
        const rightHeight = this.getHeight(root.right);

        if (Math.abs(leftHeight - rightHeight) > 1) {
            return false;
        }

        return isBalanced(root.left) && isBalanced(root.right);
    }

    getHeight(node) {
        if (!node) {
            return 0;
        }

        return (
            Math.max(this.getHeight(node.left), this.getHeight(node.right)) + 1
        );
    }
}

console.log('========= Q79 =========');
const balancedTreeRoot = new TreeNode(1);
balancedTreeRoot.left = new TreeNode(2);
balancedTreeRoot.right = new TreeNode(3);
balancedTreeRoot.left.left = new TreeNode(4);
balancedTreeRoot.left.right = new TreeNode(5);
balancedTreeRoot.left.left.left = new TreeNode(6);

const balancedTree = new BalancedBinaryTree();
console.log(
    `Is the binary tree balanced? ${balancedTree.isBalanced(balancedTreeRoot)}`
);
console.log('\n');

/*
 * Q80.
 * The ancient Egyptians used to express fractions as a sum of several terms
 * where each numerator is one. For example, 4 / 13 can be represented as 1 / 4
 * + 1 / 18 + 1 / 468.
 * Create an algorithm to turn an ordinary fraction a / b, where a < b, into an
 * Egyptian fraction.
 */
function convertToEgyptianFraction(numerator, denominator) {
    let egyptianFractions = [];

    while (numerator !== 0) {
        // Calculate the largest unit fraction that is less than or equal to the target
        // fraction
        let unitDenominator = Math.floor(
            (denominator + numerator - 1) / numerator
        );
        egyptianFractions.push(`1/${unitDenominator} `);

        numerator = numerator * unitDenominator - denominator;
        denominator *= unitDenominator;
    }

    return egyptianFractions;
}

console.log('========= Q80 =========');
const numerator = 4;
const denominator = 13;
console.log(
    `Egyptian fraction of ${numerator}/${denominator}: ${convertToEgyptianFraction(
        numerator,
        denominator
    )}`
);
console.log(`\n`);

/*
 * Q81.
 * The transitive closure of a graph is a measure of which vertices are
 * reachable from other vertices. It can be represented as a matrix M, where
 * M[i][j] == 1 if there is a path between vertices i and j, and otherwise 0.
 * For example, suppose we are given the following graph in adjacency list form:
 * graph = [
 * [0, 1, 3],
 * [1, 2],
 * [2],
 * [3]
 * ]
 * The transitive closure of this graph would be:
 * [1, 1, 1, 1]
 * [0, 1, 1, 0]
 * [0, 0, 1, 0]
 * [0, 0, 0, 1]
 * Given a graph, find its transitive closure.
 */
function findTransitiveClosure(graph) {
    const n = graph.length;
    let closure = Array.from(Array(n), () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        closure[i][i] = 1; // Each vertex is reachable from itself
        for (let j of graph[i]) {
            closure[i][j] = 1; // Mark the adjacent vertices as reachable
        }
    }

    // Apply the Warshall's algorithm to compute transitive closure
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                closure[i][j] =
                    closure[i][j] || (closure[i][k] && closure[k][j]);
            }
        }
    }

    return closure;
}

console.log('========= Q81 =========');
const graph = [
    [0, 1, 3],
    [1, 2],
    [2],
    [3]
];
console.log(`Transitive closure of ${graph}: `);
console.log(findTransitiveClosure(graph));
console.log(`\n`);

/*
 * Q82.
 * Given an array of integers out of order, determine the bounds of the smallest
 * window that must be sorted in order for the entire array to be sorted. For
 * example, given [3, 7, 5, 6, 9], you should return (1, 3).
 */
function findBounds(arr) {
    let left = 0;
    let right = arr.length - 1;

    while (left < arr.length - 1 && arr[left] <= arr[left + 1]) {
        left++;
    }

    if (left === arr.length - 1) {
        return [-1, -1];
    }

    while (right > 0 && arr[right] >= arr[right - 1]) {
        right--;
    }

    // Find the minimum and maximum elements within the subarray
    let min = Number.MAX_SAFE_INTEGER;
    let max = Number.MIN_SAFE_INTEGER;
    for (let i = left; i <= right; i++) {
        min = Math.min(min, arr[i]);
        max = Math.max(max, arr[i]);
    }

    // Expand the bounds to include any elements that need to be sorted
    while (left > 0 && arr[left - 1] > min) {
        left--;
    }

    while (right < arr.length - 1 && arr[right + 1] < max) {
        right++;
    }

    return [left, right];
}

console.log('========= Q82 =========');
const arr = [3, 7, 5, 6, 9];
console.log(`Bounds of the smallest window to be sorted: ${findBounds(arr)}`);
console.log(`\n`);

/*
 * Q83.
 * In Ancient Greece, it was common to write text with the first line going left
 * to right, the second line going right to left, and continuing to go back and
 * forth. This style was called "boustrophedon".
 * Given a binary tree, write an algorithm to print the nodes in boustrophedon
 * order.
 * For example, given the following tree:
 * "       1           "
 * "    /     \        "
 * "  2         3      "
 * " / \       / \     "
 * "4   5     6   7    "
 * You should return [1, 3, 2, 4, 5, 6, 7].
 */
function printBoustrophedon(root) {
    if (!root) {
        return;
    }

    let queue = [];
    queue.push(root);
    let leftToRight = true;
    let printString = '';

    while (queue.length > 0) {
        let levelNode = [];
        let levelSize = queue.length;

        for (let i = 0; i < levelSize; i++) {
            let node = queue.shift();

            levelNode.push(node.val);

            if (node.left) {
                queue.push(node.left);
            }

            if (node.right) {
                queue.push(node.right);
            }
        }

        if (!leftToRight) {
            levelNode.reverse();
        }

        for (const val of levelNode) {
            printString += `${val} `;
        }

        leftToRight = !leftToRight;
    }

    console.log(printString);
}

console.log('========= Q83 =========');
const boustrophedonTree = new TreeNode(1);
boustrophedonTree.left = new TreeNode(2);
boustrophedonTree.right = new TreeNode(3);
boustrophedonTree.left.left = new TreeNode(4);
boustrophedonTree.left.right = new TreeNode(5);
boustrophedonTree.right.left = new TreeNode(6);
boustrophedonTree.right.right = new TreeNode(7);

console.log(`Boustrophedon order of the tree:`);
printBoustrophedon(boustrophedonTree);
console.log(`\n`);

/*
 * Q84.
 * Huffman coding is a method of encoding characters based on their frequency.
 * Each letter is assigned a variable-length binary string, such as 0101 or
 * 111110, where shorter lengths correspond to more common letters. To
 * accomplish this, a binary tree is built such that the path from the root to
 * any leaf uniquely maps to a character. When traversing the path, descending
 * to a left child corresponds to a 0 in the prefix, while descending right
 * corresponds to 1.
 * Here is an example tree (note that only the leaf nodes have letters):
 *
 * "        *          "
 * "      /   \        "
 * "    *       *      "
 * "   / \     / \     "
 * "  *   a   t   *    "
 * " /             \   "
 * "c               s  "
 * With this encoding, cats would be represented as 0000110111.
 * Given a dictionary of character frequencies, build a Huffman tree, and use it
 * to determine a mapping between characters and their encoded binary strings.
 */
function buildHuffmanTree(root) {
    const result = { value: '' };
    traverseHuffmanTree(root, '', result);
    return result.value;
}

function traverseHuffmanTree(node, currentCode, result) {
    if (!node) {
        return;
    }

    if (!node.left && !node.right) {
        result.value += currentCode;
        return;
    }

    traverseHuffmanTree(node.left, currentCode + '0', result);
    traverseHuffmanTree(node.right, currentCode + '1', result);
}

console.log('========= Q84 =========');
const huffmanTree = new TreeNode('*');
huffmanTree.left = new TreeNode('*');
huffmanTree.right = new TreeNode('*');
huffmanTree.left.left = new TreeNode('*');
huffmanTree.left.right = new TreeNode('a');
huffmanTree.right.left = new TreeNode('t');
huffmanTree.right.right = new TreeNode('*');
huffmanTree.left.left.left = new TreeNode('c');
huffmanTree.right.right.right = new TreeNode('s');

console.log(`Huffman tree: ${buildHuffmanTree(huffmanTree)}`);
console.log(`\n`);

/*
 * Q85.
 * MegaCorp wants to give bonuses to its employees based on how many lines of
 * codes they have written. They would like to give the smallest positive amount
 * to each worker consistent with the constraint that if a developer has written
 * more lines of code than their neighbor, they should receive more money.
 * Given an array representing a line of seats of employees at MegaCorp,
 * determine how much each one should get paid.
 * For example, given [10, 40, 200, 1000, 60, 30], you should return [1, 2, 3,
 * 4, 2, 1].
 */
function calculateBonuses(linesOfCode) {
    const n = linesOfCode.length;
    let bonuses = new Array(n).fill(1);

    for (let i = 1; i < n; i++) {
        if (linesOfCode[i] > linesOfCode[i - 1]) {
            // If current employee has more lines of code than the previous one, increase
            // their bonus
            bonuses[i] = bonuses[i - 1] + 1;
        }
    }

    for (let i = n - 2; i >= 0; i--) {
        if (linesOfCode[i] > linesOfCode[i + 1]) {
            // If current employee has more lines of code than the next one, increase
            // their bonus
            bonuses[i] = Math.max(bonuses[i], bonuses[i + 1] + 1);
        }
    }

    return bonuses;
}

console.log('========= Q85 =========');
const linesOfCode = [10, 40, 200, 1000, 60, 30];
console.log(`Bonuses: ${calculateBonuses(linesOfCode)}`);
console.log(`\n`);

/*
 * Q86.
 * A step word is formed by taking a given word, adding a letter, and
 * anagramming the result. For example, starting with the word "APPLE", you can
 * add an "A" and anagram to get "APPEAL".
 * Given a dictionary of words and an input word, create a function that returns
 * all valid step words.
 */
function areAnagrams(word1, word2) {
    const sortedWord1 = word1.split('').sort().join('');
    const sortedWord2 = word2.split('').sort().join('');
    return sortedWord1 === sortedWord2;
}

function findStepWords(inputWord, dictionary) {
    const stepWords = [];

    for (const word of dictionary) {
        for (let i = 65; i <= 90; i++) {
            const char = String.fromCharCode(i);
            const newWord = inputWord + char;

            if (areAnagrams(newWord, word)) {
                stepWords.push(word);
                break;
            }
        }
    }

    return stepWords;
}

console.log('========= Q86 =========');
const inputWord = 'APPLE';
const dictionaryForAnagram = ['APPEAL', 'PEAR', 'PLEA', 'LEAP'];
const stepWords = findStepWords(inputWord, dictionaryForAnagram);
console.log(`Step words: ${stepWords}`);
console.log(`\n`);

/*
 * Q87.
 * You are given an string representing the initial conditions of some dominoes.
 * Each element can take one of three values:
 * L, meaning the domino has just been pushed to the left,
 * R, meaning the domino has just been pushed to the right, or
 * ., meaning the domino is standing still.
 * Determine the orientation of each tile when the dominoes stop falling. Note
 * that if a domino receives a force from the left and right side
 * simultaneously, it will remain upright.
 * For example, given the string .L.R....L, you should return LL.RRRLLL.
 * Given the string ..R...L.L, you should return ..RR.LLLL.
 */
function orientDominos(dominoes) {
    let orientations = dominoes.split('');
    const n = orientations.length;
    let forces = Array(n).fill(0);

    let force = 0; //Cumulative force
    for (let i = 0; i < n; i++) {
        if (orientations[i] === 'R') {
            force = n; // Max force if pushed from the right
        } else if (orientations[i] === 'L') {
            force = 0; // Reset force if pushed from the left
        } else {
            force = Math.max(force - 1, 0); // Decrease force gradually
        }

        forces[i] += force;
    }

    force = 0;
    for (let i = n - 1; i >= 0; i--) {
        if (orientations[i] === 'L') {
            force = n; // Max force if pushed from the left
        } else if (orientations[i] === 'R') {
            force = 0; // Reset force if pushed from the right
        } else {
            force = Math.max(force - 1, 0); // Decrease force gradually
        }

        forces[i] -= force;
    }

    let result = '';
    for (const force of forces) {
        if (force > 0) {
            result += 'R';
        } else if (force < 0) {
            result += 'L';
        } else {
            result += '.';
        }
    }
    return result;
}

console.log('========= Q87 =========');
const dominoes1 = '.L.R....L';
const dominoes2 = '..R...L.L';
console.log(`Oriented dominoes1: ${orientDominos(dominoes1)}`); // Output: LL.RRRLLL
console.log(`Oriented dominoes1: ${orientDominos(dominoes2)}`); // Output: ..RR.LLLL
console.log(`\n`);

/*
 * Q88.
 * A fixed point in an array is an element whose value is equal to its index.
 * Given a sorted array of distinct elements, return a fixed point, if one
 * exists. Otherwise, return False.
 * For example, given [-6, 0, 2, 40], you should return 2. Given [1, 5, 7, 8],
 * you should return False.
 */
function findFixedPoint(nums) {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);

        if (nums[mid] === mid) {
            return mid;
        } else if (nums[mid] < mid) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return false;
}

console.log('========= Q88 =========');
const nums1 = [-6, 0, 2, 40];
const nums2 = [1, 5, 7, 8];
console.log(`Fixed point of nums1: ${findFixedPoint(nums1)}`); // Output: 2
console.log(`Fixed point of nums2: ${findFixedPoint(nums2)}`); // Output: false
console.log(`\n`);

/*
 * Q89.
 * UTF-8 is a character encoding that maps each symbol to one, two, three, or
 * four bytes.
 * For example, the Euro sign, €, corresponds to the three bytes 11100010
 * 10000010 10101100. The rules for mapping characters are as follows:
 * For a single-byte character, the first bit must be zero.
 * For an n-byte character, the first byte starts with n ones and a zero. The
 * other n - 1 bytes all start with 10.
 * Visually, this can be represented as follows.
 * Bytes | Byte format
 * -----------------------------------------------
 * 1 | 0xxxxxxx
 * 2 | 110xxxxx 10xxxxxx
 * 3 | 1110xxxx 10xxxxxx 10xxxxxx
 * 4 | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 * Write a program that takes in an array of integers representing byte values,
 * and returns whether it is a valid UTF-8 encoding.
 */
function validUtf8(data) {
    let remainingBytes = 0;

    for (const num of data) {
        if (remainingBytes === 0) {
            if (num >> 5 === 0b110) {
                remainingBytes = 1;
            } else if (num >> 4 === 0b1110) {
                remainingBytes = 2;
            } else if (num >> 3 === 0b11110) {
                remainingBytes = 3;
            } else if (num >> 7 !== 0) {
                return false;
            }
        } else {
            if (num >> 6 !== 0b10) {
                return false;
            }
            remainingBytes--;
        }
    }

    return remainingBytes === 0;
}

console.log('========= Q89 =========');
const data1 = [197, 130, 1]; //11000101 10000010 00000001
const data2 = [235, 140, 4]; //11101011 10001100 00000100
console.log(`Is valid UTF-8 of data1: ${validUtf8(data1)}`); // Output: true
console.log(`Is valid UTF-8 of data2: ${validUtf8(data2)}`); // Output: false
console.log(`\n`);

/*
 * Q90.
 * Given an integer N, construct all possible binary search trees with N nodes.
 */
function generateBSTs(n) {
    return generateTrees(1, n);
}

function generateTrees(start, end) {
    let trees = [];

    if (start > end) {
        trees.push(null);
        return trees;
    }

    for (let i = start; i <= end; i++) {
        let leftTrees = generateTrees(start, i - 1);
        let rightTrees = generateTrees(i + 1, end);

        for (let left of leftTrees) {
            for (let right of rightTrees) {
                let root = new TreeNode(i);
                root.left = left;
                root.right = right;
                trees.push(root);
            }
        }
    }

    return trees;
}

console.log('========= Q90 =========');
const nForBinarySearchTree = 3;
console.log(`All possible BSTs of ${nForBinarySearchTree} nodes: `);
const possibleBSTs = generateBSTs(nForBinarySearchTree);
possibleBSTs.forEach((bst) => {
    console.log(bst);
});
console.log(`\n`);

/*
 * Q91.
 * A classroom consists of N students, whose friendships can be represented in
 * an adjacency list. For example, the following describes a situation where 0 is
 * friends with 1 and 2, 3 is friends with 6, and so on.
 * {0: [1, 2],
 * 1: [0, 5],
 * 2: [0],
 * 3: [6],
 * 4: [],
 * 5: [1],
 * 6: [3]}
 * Each student can be placed in a friend group, which can be defined as the
 * transitive closure of that student's friendship relations. In other words,
 * this is the smallest set such that no student in the group has any friends
 * outside this group. For the example above, the friend groups would be {0, 1,
 * 2, 5}, {3, 6}, {4}.
 * Given a friendship list such as the one above, determine the number of friend
 * groups in the class.
 */
function countFriendGroups(adjacencyList) {
    const n = adjacencyList.length;
    let visited = new Array(n).fill(false);
    let count = 0;

    for (let i = 0; i < n; i++) {
        if (!visited[i]) {
            dfsToFindFriendGroup(adjacencyList, i, visited);
            count++;
        }
    }

    return count;
}

function dfsToFindFriendGroup(adjacencyList, node, visited) {
    visited[node] = true;

    for (let neighbor of adjacencyList[node]) {
        if (!visited[neighbor]) {
            dfsToFindFriendGroup(adjacencyList, neighbor, visited);
        }
    }
}

console.log('========= Q91 =========');
const adjacencyList = [
    [1, 2],
    [0, 5],
    [0],
    [6],
    [],
    [1],
    [3]
];
console.log(`Number of friend groups: ${countFriendGroups(adjacencyList)}`); // Output: 3
console.log(`\n`);

/*
 * Q92.
 * Given an undirected graph, determine if it contains a cycle.
 */
class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.adjacencyList = Array(vertices)
            .fill(null)
            .map(() => []);
    }

    addEdge(src, dest) {
        this.adjacencyList[src].push(dest);
        this.adjacencyList[dest].push(src);
    }

    containCycle() {
        let visited = new Array(this.vertices).fill(false);

        for (let i = 0; i < this.vertices; i++) {
            if (!visited[i] && this.hasCycle(i, visited, -1)) {
                return true;
            }
        }

        return false;
    }

    hasCycle(vertex, visited, parent) {
        visited[vertex] = true;

        for (let neighbor of this.adjacencyList[vertex]) {
            if (!visited[neighbor]) {
                if (this.hasCycle(neighbor, visited, vertex)) {
                    return true;
                }
            } else if (neighbor !== parent) {
                // If the neighbor is already visited and not the parent of the current vertex,
                // a cycle is found.
                return true;
            }
        }

        return false;
    }
}

console.log('========= Q92 =========');
const graphWithCycle = new Graph(5);
graphWithCycle.addEdge(0, 1);
graphWithCycle.addEdge(1, 2);
graphWithCycle.addEdge(2, 3);
graphWithCycle.addEdge(3, 4);
graphWithCycle.addEdge(4, 1);
console.log(graphWithCycle.adjacencyList);
console.log(`Does this graph contain cycle: ${graphWithCycle.containCycle()}`); // Output: true
console.log(`\n`);

/*
 * Q93.
 * Given an array of integers, determine whether it contains a Pythagorean
 * triplet. Recall that a Pythagorean triplet (a, b, c) is defined by the
 * equation a2+ b2= c2.
 */
function containsPythagoreanTriplet(nums) {
    const n = nums.length;

    for (let i = 0; i < n; i++) {
        nums[i] = nums[i] * nums[i];
    }

    nums.sort((a, b) => a - b);

    for (let i = 0; i < n - 2; i++) {
        for (let j = i + 1; j < n - 1; j++) {
            const squareSum = nums[i] + nums[j];
            if (binarySearch(nums, squareSum, j + 1, n - 1)) {
                return true;
            }
        }
    }

    return false;
}

function binarySearch(nums, target, left, right) {
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] === target) {
            return true;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return false;
}

console.log('========= Q93 =========');
const numsForPythagorean = [3, 1, 4, 6, 5];
console.log(
    `Array contain a Pythagorean triplet: ${containsPythagoreanTriplet(
        numsForPythagorean
    )}`
); // Output: true
console.log(`\n`);

/*
 * Q94.
 * A regular number in mathematics is defined as one which evenly divides some
 * power of 60. Equivalently, we can say that a regular number is one whose only
 * prime divisors are 2, 3, and 5.
 * These numbers have had many applications, from helping ancient Babylonians
 * keep time to tuning instruments according to the diatonic scale.
 * Given an integer N, write a program that returns, in order, the first N
 * regular numbers.
 */
function getRegularNumbers(n) {
    let regularNumbers = [];
    regularNumbers.push(1);
    let i2 = 0,
        i3 = 0,
        i5 = 0;

    while (regularNumbers.length < n) {
        let nextMultipleOf2 = regularNumbers[i2] * 2;
        let nextMultipleOf3 = regularNumbers[i3] * 3;
        let nextMultipleOf5 = regularNumbers[i5] * 5;

        let nextRegularNumber = Math.min(
            nextMultipleOf2,
            nextMultipleOf3,
            nextMultipleOf5
        );
        regularNumbers.push(nextRegularNumber);

        if (nextRegularNumber === nextMultipleOf2) {
            i2++;
        }
        if (nextRegularNumber === nextMultipleOf3) {
            i3++;
        }
        if (nextRegularNumber === nextMultipleOf5) {
            i5++;
        }
    }

    return regularNumbers;
}

console.log('========= Q94 =========');
console.log(`First 10 regular numbers: ${getRegularNumbers(10)}`); // Output: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12]
console.log(`\n`);

/*
 * Q95.
 * On a mysterious island there are creatures known as Quxes which come in three
 * colors: red, green, and blue. One power of the Qux is that if two of them are
 * standing next to each other, they can transform into a single creature of the
 * third color.
 * Given N Quxes standing in a line, determine the smallest number of them
 * remaining after any possible sequence of such transformations.
 * For example, given the input ['R', 'G', 'B', 'G', 'B'], it is possible to end
 * up with a single Qux through the following steps:
 * "        Arrangement       |   Change    "
 * "----------------------------------------"
 * "['R', 'G', 'B', 'G', 'B'] | (R, G) -> B "
 * "['B', 'B', 'G', 'B']      | (B, G) -> R "
 * "['B', 'R', 'B']           | (R, B) -> G "
 * "['B', 'G']                | (B, G) -> R "
 * "['R']                     |             "
 */
function getRemainingQuxes(quxes) {
    let stack = [];

    for (const qux of quxes) {
        if (stack.length > 0 && stack[stack.length - 1] !== qux) {
            stack.pop(); // Quxes transform into the third color
        } else {
            stack.push(qux);
        }
    }
    return stack.length;
}

console.log('========= Q95 =========');
const quxes = ['R', 'G', 'B', 'G', 'B'];
console.log(`Remaining quxes: ${getRemainingQuxes(quxes)}`); // Output: 1
console.log(`\n`);

/*
 * Q96.
 * A girl is walking along an apple orchard with a bag in each hand. She likes
 * to pick apples from each tree as she goes along, but is meticulous about not
 * putting different kinds of apples in the same bag.
 * Given an input describing the types of apples she will pass on her path, in
 * order, determine the length of the longest portion of her path that consists
 * of just two types of apple trees.
 * For example, given the input [2, 1, 2, 3, 3, 1, 3, 5], the longest portion
 * will involve types 1 and 3, with a length of four.
 */
function longestTwoAppleTrees(trees) {
    const n = trees.length;

    if (n < 3) {
        return n;
    }

    let maxLen = 0;
    let treeCounts = new Map();
    let left = 0;
    let right = 0;

    while (right < n) {
        treeCounts.set(trees[right], (treeCounts.get(trees[right]) || 0) + 1);

        while (treeCounts.size > 2) {
            treeCounts.set(trees[left], treeCounts.get(trees[left]) - 1);
            if (treeCounts.get(trees[left]) === 0) {
                treeCounts.delete(trees[left]);
            }
            left++;
        }

        maxLen = Math.max(maxLen, right - left + 1);
        right++;
    }

    return maxLen;
}

console.log('========= Q96 =========');
const appleTrees = [2, 1, 2, 3, 3, 1, 3, 5];
console.log(`Longest portion of path: ${longestTwoAppleTrees(appleTrees)}`); // Output: 4
console.log(`\n`);

/*
 * Q97.
 * On election day, a voting machine writes data in the form (voter_id,
 * candidate_id) to a text file. Write a program that reads this file as a
 * stream and returns the top 3 candidates at any given time. If you find a
 * voter voting more than once, report this as fraud.
 */
//NOT tested
class Candidate {
    constructor(id, count) {
        this.id = id;
        this.count = count;
    }

    getId() {
        return this.id;
    }

    getCount() {
        return this.count;
    }
}

console.log('========= Q97 =========');
const fs = require('fs');
const filePath = 'voting_data.txt';
let voteCounts = new Map();
let topCandidates = [];

try {
    const data = fs.readFileSync(filePath, 'utf8');
    const lines = data.split('\n');

    for (const line of lines) {
        const [voterId, candidateId] = line.split(',');

        if (voteCounts.has(voterId)) {
            console.log(
                `Voter fraud detected: Voter ${voterId} voted more than once`
            );
            continue;
        }

        voteCounts.set(voterId, candidateId);
        let count = voteCounts.get(candidateId)++ || 1;
        voteCounts.set(candidateId, count);

        let candidate = new Candidate(candidateId, count);
        topCandidates.push(candidate);
        topCandidates.sort((a, b) => b.count - a.count);

        if (topCandidates.length > 3) {
            topCandidates.pop();
        }

        console.log(`Top 3 candidates:`);
        for (let i = 0; i < 3; i++) {
            console.log(
                `Candidate ${topCandidates[i].getId()}: ${topCandidates[
                    i
                ].getCount()} votes`
            );
        }
    }
} catch (err) {
    console.error(err);
}
console.log(`\n`);

/*
 * Q98.
 * Given a clock time in hh:mm format, determine, to the nearest degree, the
 * angle between the hour and the minute hands.
 * Bonus: When, during the course of a day, will the angle be zero?
 */
function calculateAngle(time) {
    const parts = time.split(':');
    const hour = parseInt(parts[0]);
    const minute = parseInt(parts[1]);

    const hourAngle = (hour % 12) * 30 + minute * 0.5;
    const minuteAngle = minute * 6;
    const angle = Math.abs(hourAngle - minuteAngle);

    return Math.min(angle, 360 - angle);
}

console.log('========= Q98 =========');
const time = '10:45';
console.log(
    `Angle between hour and minute hands: ${calculateAngle(time)} degrees`
); // Output: 52.5

// Bonus: Find when the angle is zero
let count = 0;
for (let hour = 0; hour < 12; hour++) {
    for (let minute = 0; minute < 60; minute++) {
        const currAngle = calculateAngle(`${hour}:${minute}`);
        if (currAngle === 0) {
            count++;
        }
    }
}
console.log(`Total instances where angle is zero: ${count}`);
console.log(`\n`);

/*
 * Q99.
 * Given a linked list, remove all consecutive nodes that sum to zero. Print out
 * the remaining nodes.
 * For example, suppose you are given the input 3 -> 4 -> -7 -> 5 -> -6 -> 6. In
 * this case, you should first remove 3 -> 4 -> -7, then -6 -> 6, leaving only
 * 5.
 */
function removeZeroSumSublists(head) {
    let dummy = new ListNode(0);
    dummy.next = head;

    // Create a prefix sum map to track cumulative sums and their corresponding
    // nodes
    let prefixSumMap = new Map();
    let prefixSum = 0;
    let curr = dummy;

    while (curr) {
        prefixSum += curr.val;

        if (prefixSumMap.has(prefixSum)) {
            // Remove the nodes between the previous occurrence of the prefix sum and the
            // current node
            let prev = prefixSumMap.get(prefixSum).next;
            let sum = prefixSum + prev.val;

            while (prev !== curr) {
                prefixSumMap.delete(sum);
                prev = prev.next;
                sum += prev.val;
            }

            prefixSumMap.get(prefixSum).next = curr.next;
        } else {
            prefixSumMap.set(prefixSum, curr);
        }

        curr = curr.next;
    }

    return dummy.next;
}

console.log('========= Q99 =========');
let headToRemoveConsecutiveSumZero = new ListNode(3);
headToRemoveConsecutiveSumZero.next = new ListNode(4);
headToRemoveConsecutiveSumZero.next.next = new ListNode(-7);
headToRemoveConsecutiveSumZero.next.next.next = new ListNode(5);
headToRemoveConsecutiveSumZero.next.next.next.next = new ListNode(-6);
headToRemoveConsecutiveSumZero.next.next.next.next.next = new ListNode(6);

console.log(`Original Linked List:`);
printLinkedList(headToRemoveConsecutiveSumZero);

const updatedHead = removeZeroSumSublists(headToRemoveConsecutiveSumZero);
console.log(`Updated Linked List:`);
printLinkedList(updatedHead);
console.log(`\n`);

/*
 * Q100.
 * Given a binary search tree, find the floor and ceiling of a given integer.
 * The floor is the highest element in the tree less than or equal to an
 * integer, while the ceiling is the lowest element in the tree greater than or
 * equal to an integer.
 * If either value does not exist, return None.
 */
function insert(root, val) {
    if (!root) {
        return new TreeNode(val);
    }

    if (val < root.val) {
        root.left = insert(root.left, val);
    } else {
        root.right = insert(root.right, val);
    }

    return root;
}

function findFloor(root, target) {
    if (!root) {
        return null;
    }

    if (root.val === target) {
        return root.val;
    }

    if (root.val > target) {
        return findFloor(root.left, target);
    }

    const floor = findFloor(root.right, target);
    return floor && floor <= target ? floor : root.val;
}

function findCeiling(root, target) {
    if (!root) {
        return null;
    }

    if (root.val === target) {
        return root.val;
    }

    if (root.val < target) {
        return findCeiling(root.right, target);
    }

    const ceiling = findCeiling(root.left, target);
    return ceiling && ceiling >= target ? ceiling : root.val;
}

console.log('========= Q100 =========');
let treeToFindFloorOrCeiling;
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 8);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 4);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 12);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 2);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 6);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 10);
treeToFindFloorOrCeiling = insert(treeToFindFloorOrCeiling, 14);

const targetForFloorAndCeiling = 7;
const floor = findFloor(treeToFindFloorOrCeiling, targetForFloorAndCeiling);
const ceiling = findCeiling(treeToFindFloorOrCeiling, targetForFloorAndCeiling);

console.log(`Target: ${targetForFloorAndCeiling}`);
console.log(`Floor: ${floor ? floor : 'None'}`);
console.log(`Ceiling: ${ceiling ? ceiling : 'None'}`);
console.log(`\n`);

/*
 * Q101.
 * Write an algorithm that finds the total number of set bits in all integers
 * between 1 and N.
 */
function countSetBits(n) {
    let count = 0;

    for (let i = 1; i <= n; i++) {
        count += countSetBitsUtil(i);
    }

    return count;
}

function countSetBitsUtil(num) {
    let count = 0;

    while (num > 0) {
        if ((num & 1) === 1) {
            count++;
        }

        num >>= 1;
    }

    return count;
}

console.log('========= Q101 =========');
const numToGetNumOfBits = 10;
console.log(`total number of set bits in between 1 and ${numToGetNumOfBits}: ${countSetBits(numToGetNumOfBits)}`);
console.log(`\n`);

/*
 * Q102.
 * Given a array that's sorted but rotated at some unknown pivot, in which all
 * elements are distinct, find a "peak" element in O(log N) time.
 * An element is considered a peak if it is greater than both its left and right
 * neighbors. It is guaranteed that the first and last elements are lower than
 * all others.
 */
function findPeakElement(nums) {
    let left = 0;
    let right = nums.length - 1;

    while (left < right) {
        const mid = left + Math.floor((right - left) / 2);

        if (nums[mid] > nums[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left - 1;
}

console.log('========= Q102 =========');
const numsToFindPeak = [5, 6, 7, 8, 9, 10, 1, 2, 3, 4];
const peakElementIndex = findPeakElement(numsToFindPeak);
console.log(`Peak element: ${numsToFindPeak[peakElementIndex]}`);
console.log(`\n`);

/*
 * Q103.
 * You are given a 2 x N board, and instructed to completely cover the board
 * with the following shapes:
 * Dominoes, or 2 x 1 rectangles.
 * Trominoes, or L-shapes.
 * For example, if N = 4, here is one possible configuration, where A is a
 * domino, and B and C are trominoes.
 * A B B C
 * A B C C
 * Given an integer N, determine in how many ways this task is possible.
 */
function countWaysToCoverBoard(N) {
    if (N === 0) {
        return 1;
    }

    let dp = new Array(N + 1).fill(0);
    dp[0] = 1;
    dp[1] = 1;

    for (let i = 2; i <= N; i++) {
        //dp[i-1]: the number of ways to cover the board by extending the previous width by a 2x1 domino
        //dp[i-2]: the number of ways to cover the board by extending the previous width by a 2x2 tromino
        //2*dp[i-2]: the number of ways to cover the board by placing two 1x2 trominoes vertically
        dp[i] = dp[i - 1] + dp[i - 2] + 2 * dp[i - 2];
    }

    return dp[N]; //1 1 4 7 19 by increasing N by 1
}

console.log('========= Q103 =========');
const colsForBoard = 4;
const ways = countWaysToCoverBoard(colsForBoard);
console.log(`Number of ways to cover the board: ${ways}`);
console.log(`\n`);

/*
 * Q104.
 * In linear algebra, a Toeplitz matrix is one in which the elements on any
 * given diagonal from top left to bottom right are identical.
 * Here is an example:
 * 1 2 3 4 8
 * 5 1 2 3 4
 * 4 5 1 2 3
 * 7 4 5 1 2
 * Write a program to determine whether a given input is a Toeplitz matrix.
 */
function isToeplitzMatrix(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;

    for (let i = 0; i < rows - 1; i++) {
        for (let j = 0; j < cols - 1; j++) {
            if (matrix[i][j] !== matrix[i + 1][j + 1]) {
                return false;
            }
        }
    }

    return true;
}

console.log('========= Q104 =========');
const toeplitzMatrix = [
    [1, 2, 3, 4, 8],
    [5, 1, 2, 3, 4],
    [4, 5, 1, 2, 3],
    [7, 4, 5, 1, 2]
];
const isToeplitz = isToeplitzMatrix(toeplitzMatrix);
console.log(`Is Toeplitz matrix: ${isToeplitz}`);
console.log(`\n`);

/*
* Q105.
* Consider the following scenario: there are N mice and N holes placed at
* integer points along a line. Given this, find a method that maps mice to
* holes such that the largest number of steps any mouse takes is minimized.
* Each move consists of moving one mouse one unit to the left or right, and
* only one mouse can fit inside each hole.
* For example, suppose the mice are positioned at [1, 4, 9, 15], and the holes
* are located at [10, -5, 0, 16]. In this case, the best pairing would require
* us to send the mouse at 1 to the hole at -5, so our function should return 6.
*/
function minimizeSteps(mice, holes) {
    mice.sort((a, b) => a - b);
    holes.sort((a, b) => a - b);

    let maxSteps = 0;

    for (let i = 0; i < mice.length; i++) {
        let steps = Math.abs(mice[i] - holes[i]);
        maxSteps = Math.max(maxSteps, steps);
    }

    return maxSteps;
}

console.log('========= Q105 =========');
const mice = [1, 4, 9, 15];
const holes = [10, -5, 0, 16];

const minMouseSteps = minimizeSteps(mice, holes);
console.log(`The minimum number of steps required is: ${minMouseSteps}`);
console.log(`\n`);

/*
* Q106.
* The United States uses the imperial system of weights and measures, which
* means that there are many different, seemingly arbitrary units to measure
* distance. There are 12 inches in a foot, 3 feet in a yard, 22 yards in a
* chain, and so on.
* Create a data structure that can efficiently convert a certain quantity of
* one unit to the correct amount of any other unit. You should also allow for
* additional units to be added to the system.
*/
class UnitConverter {
    #conversionGraph;

    constructor() {
        this.#conversionGraph = new Map();
    }

    addUnitConversion(fromUnit, toUnit, conversionRate) {
        this.#conversionGraph.set(fromUnit, new Map());
        this.#conversionGraph.set(toUnit, new Map());

        this.#conversionGraph.get(fromUnit).set(toUnit, conversionRate);
        this.#conversionGraph.get(toUnit).set(fromUnit, 1 / conversionRate);
    }

    convert(quantity, fromUnit, toUnit) {
        if (fromUnit === toUnit) {
            return quantity;
        }

        if (!this.#conversionGraph.has(fromUnit) || !this.#conversionGraph.has(toUnit)) {
            throw new Error(`Conversion not defined between ${fromUnit} and ${toUnit}`);
        }

        let distances = new Map();
        distances.set(fromUnit, 1);

        let queue = [fromUnit];

        while (queue.length > 0) {
            let currentUnit = queue.shift();
            let currentDistance = distances.get(currentUnit);

            if (currentUnit === toUnit) {
                return quantity * currentDistance;
            }

            let neighbors = this.#conversionGraph.get(currentUnit);
            for (let neighbor of neighbors.keys()) {
                if (!distances.has(neighbor)) {
                    let neighborDistance = currentDistance * neighbors.get(neighbor);
                    distances.set(neighbor, neighborDistance);
                    queue.push(neighbor);
                }
            }
        }

        throw new Error(`Conversion not possible from ${fromUnit} to ${toUnit}`);
    }
}

console.log('========= Q106 =========');
const unitConverter = new UnitConverter();
unitConverter.addUnitConversion('inch', 'foot', 1 / 12);
unitConverter.addUnitConversion('foot', 'yard', 1 / 3);
unitConverter.addUnitConversion('yard', 'chain', 1 / 22);

const quantity = 1000;
const fromUnit = 'inch';
const toUnit = 'yard';
const convertedResult = unitConverter.convert(quantity, fromUnit, toUnit);
console.log(`${quantity} ${fromUnit} = ${convertedResult} ${toUnit}`);
console.log(`\n`);

/*
* Q107.
* Write a program to merge two binary trees. Each node in the new tree should
* hold a value equal to the sum of the values of the corresponding nodes of the
* input trees.
* If only one input tree has a node in a given position, the corresponding node
* in the new tree should match that input node.
*/
function mergeTrees(t1, t2) {
    if (!t1 && !t2) {
        return null;
    }

    let val1 = t1 ? t1.val : 0;
    let val2 = t2 ? t2.val : 0;

    let newNode = new TreeNode(val1 + val2);
    newNode.left = mergeTrees(t1 ? t1.left : null, t2 ? t2.left : null);
    newNode.right = mergeTrees(t1 ? t1.right : null, t2 ? t2.right : null);

    return newNode;
}

function printPreorder(root) {
    if (root) {
        console.log(root.val);
        printPreorder(root.left);
        printPreorder(root.right);
    }
}
console.log('========= Q107 =========');
const t1 = new TreeNode(1);
t1.left = new TreeNode(3);
t1.right = new TreeNode(2);
t1.left.left = new TreeNode(5);

const t2 = new TreeNode(2);
t2.left = new TreeNode(1);
t2.right = new TreeNode(3);
t2.left.right = new TreeNode(4);
t2.right.right = new TreeNode(7);

const mergedTree = mergeTrees(t1, t2);
printPreorder(mergedTree);
console.log(`\n`);

/*
* Q108.
* Given integers M and N, write a program that counts how many positive integer
* pairs (a, b) satisfy the following conditions:
* a + b = M
* a XOR b = N
*/
function countPairs(M, N) {
    let count = 0;

    for (let a = 1; a <= Math.floor(M / 2); a++) {
        let b = M - a;
        if ((a ^ b) === N) {
            count++;
        }
    }

    return count;
}

console.log('========= Q108 =========');
const M = 40;
const N = 20;
const pairCount = countPairs(M, N);
console.log(`Number of positive integer pairs: ${pairCount}`);
console.log(`\n`);

/*
* Q109.
* The 24 game is played as follows. You are given a list of four integers, each
* between 1 and 9, in a fixed order. By placing the operators +, -, *, and /
* between the numbers, and grouping them with parentheses, determine whether it
* is possible to reach the value 24.
* For example, given the input [5, 2, 7, 8], you should return True, since (5 *
* 2 - 7) * 8 = 24.
* Write a function that plays the 24 game.
*/
function canReach24(nums) {
    return canReachTarget(nums, 24);
}

function canReachTarget(nums, target) {
    if (nums.length === 1) {
        return nums[0] === target;
    }

    for (let i = 0; i < nums.length; i++) {
        for (let j = 0; j < nums.length; j++) {
            if (i !== j) {
                let remaining = getRemainingArray(nums, i, j);

                if (canReachTarget(applyOperators(nums[i], nums[j], remaining), target)) {
                    return true;
                }
            }
        }
    }
    return false;
}

function applyOperators(a, b, remaining) {
    let result = new Array(remaining.length + 1);

    for (let i = 0; i < remaining.length; i++) {
        result[i] = remaining[i];
    }

    result[result.length - 1] = a + b;
    if (canReachTarget(result, 24)) {
        return result;
    }

    result[result.length - 1] = a - b;
    if (canReachTarget(result, 24)) {
        return result;
    }

    result[result.length - 1] = a * b;
    if (canReachTarget(result, 24)) {
        return result;
    }

    result[result.length - 1] = a / b;
    if (canReachTarget(result, 24)) {
        return result;
    }

    return remaining;
}

function getRemainingArray(nums, i, j) {
    let remaining = new Array(nums.length - 2);
    let index = 0;

    for (let k = 0; k < nums.length; k++) {
        if (k !== i && k !== j) {
            remaining[index++] = nums[k];
        }
    }

    return remaining;
}

console.log('========= Q109 =========');
const numsFor24Game = [5, 2, 7, 8];
const canReach24Result = canReach24(numsFor24Game);
console.log(`Can reach 24: ${canReach24Result}`);
console.log(`\n`);

/*
* Q110.
* Given an array of numbers and a number k, determine if there are three
* entries in the array which add up to the specified number k. For example,
* given [20, 303, 3, 4, 25] and k = 49, return true as 20 + 4 + 25 = 49.
*/
function hasThreeSum(nums, k) {
    let set = new Set();

    for (let i = 0; i < nums.length - 2; i++) {
        let target = k - nums[i];
        set.clear();

        for (let j = i + 1; j < nums.length; j++) {
            if (set.has(target - nums[j])) {
                return true;
            }
            set.add(nums[j]);
        }
    }
    return false;
}

console.log('========= Q110 =========');
const numsForThreeSum = [20, 303, 3, 4, 25];
const kForThreeSum = 49;
const hasThreeSumResult = hasThreeSum(numsForThreeSum, kForThreeSum);
console.log(`Has three sum: ${hasThreeSumResult}`);
console.log(`\n`);
