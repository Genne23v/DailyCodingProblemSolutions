const { get } = require('http');
const { sep } = require('path');

/*
 * Q1.
 * Given an array of integers, return a new array such that each element at
 * index i of the new array is the product of all the numbers in the original
 * array except the one at i.
 * For example, if our input was [1, 2, 3, 4, 5], the expected output would be
 * [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would
 * be [2, 3, 6].
 * Follow-up: what if you can't use division?
 */
function productExceptSelf(nums) {
    const n = nums.length;
    let prefix = new Array(n);
    let suffix = new Array(n);
    let result = new Array(n);

    prefix[0] = 1;
    for (let i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] * nums[i - 1];
    }

    suffix[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] * nums[i + 1];
    }

    for (let i = 0; i < n; i++) {
        result[i] = prefix[i] * suffix[i];
    }

    return result;
}

console.log('========= Q1 =========');
const nums1 = [1, 2, 3, 4, 5];
console.log(productExceptSelf(nums1));

const nums2 = [3, 2, 1];
console.log(productExceptSelf(nums2));
console.log('\n');

/*
 * Q2.
 * Given an array of integers, find the first missing positive integer in linear
 * time and constant space. In other words, find the lowest positive integer
 * that does not exist in the array. The array can contain duplicates and
 * negative numbers as well.
 * For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0]
 * should give 3.
 * You can modify the input array in-place.
 */
function findMissingPositive(nums) {
    const n = nums.length;

    // Step 1: Ignore negative numbers and numbers greater than n
    for (let i = 0; i < n; i++) {
        if (nums[i] <= 0 || nums[i] > n) {
            nums[i] = n + 1; // Mark irrelevant numbers
        }
    }

    // Step 2: Rearrange the array using index as a hash
    for (let i = 0; i < n; i++) {
        const num = Math.abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -Math.abs(nums[num - 1]);
        }
    }

    // Step 3: Find the index of the first positive number (mismatch)
    for (let i = 0; i < n; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }

    return n + 1;
}

console.log('========= Q2 =========');
const missingInt1 = [3, 4, -1, 1];
console.log(findMissingPositive(missingInt1));

const missingInt2 = [1, 2, 0];
console.log(findMissingPositive(missingInt2));
console.log('\n');

/*
 * Q3.
 * An XOR linked list is a more memory efficient doubly linked list. Instead of
 * each node holding next and prev fields, it holds a field named both, which is
 * an XOR of the next node and the previous node. Implement an XOR linked list;
 * it has an add(element) which adds the element to the end, and a get(index)
 * which returns the node at index.
 * If using a language that has no pointers (such as Python), you can assume you
 * have access to get_pointer and dereference_pointer functions that converts
 * between nodes and memory addresses.
 */
class XORNode {
    #_data;
    #_both;

    constructor(data) {
        this.#_data = data;
    }

    get data() {
        return this.#_data;
    }

    get both() {
        return this.#_both;
    }

    set both(both) {
        this.#_both = both;
    }

    hashCode() {
        let hash = 0;
        const str = JSON.stringify(this.#_data);
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = (hash << 5) - hash + char;
            hash |= 0; // Convert to 32bit integer;
        }
        return hash;
    }
}

class XORLinkedList {
    #_head;
    #_tail;
    #_nodeMap;

    constructor() {
        this.#_nodeMap = new Map();
    }

    add(element) {
        const newNode = new XORNode(element);
        if (!this.#_head) {
            this.#_head = newNode;
            this.#_tail = newNode;
        } else {
            newNode.both = this.getPointer(this.#_tail);
            this.#_tail.both = this.#_tail.both ^ this.getPointer(newNode);
            this.#_tail = newNode;
        }
        this.#_nodeMap.set(newNode.hashCode(), newNode);
    }

    get(index) {
        let curr = this.#_head;
        let prev = null;

        for (let i = 0; i < index; i++) {
            const next = this.dereferencePointer(
                curr.both ^ this.getPointer(prev)
            );
            prev = curr;
            curr = next;
        }
        return curr;
    }

    getPointer(node) {
        return node ? node.hashCode() : 0;
    }

    dereferencePointer(pointer) {
        if (pointer === 0) {
            return null;
        }
        return this.#_nodeMap.get(pointer);
    }
}

console.log('========= Q3 =========');
const xorLinkedList = new XORLinkedList();
xorLinkedList.add(10);
xorLinkedList.add(20);
xorLinkedList.add(30);
xorLinkedList.add(40);
xorLinkedList.add(50);

const node1 = xorLinkedList.get(0);
const node2 = xorLinkedList.get(2);
const node3 = xorLinkedList.get(4);

console.log(node1.data);
console.log(node2.data);
console.log(node3.data);
console.log('\n');

/*
 * Q4.
 * Given a list of integers, write a function that returns the largest sum of
 * non-adjacent numbers. Numbers can be 0 or negative.
 * For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5,
 * 1, 1, 5] should return 10, since we pick 5 and 5.
 * Follow-up: Can you do this in O(N) time and constant space?
 */
function largestSumNonAdjacent(nums) {
    let inclusive = 0; // sum including the current element
    let exclusive = 0; // sum excluding the current element

    for (const num of nums) {
        const temp = inclusive;
        inclusive = Math.max(inclusive, exclusive + num);
        exclusive = temp;
    }

    return Math.max(inclusive, exclusive);
}

console.log('========= Q4 =========');
const numsTonFindLargestNonAdjacentSum1 = [2, 4, 6, 2, 5];
const numsTonFindLargestNonAdjacentSum2 = [5, 1, 1, 5];

console.log(largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum1));
console.log(largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum2));
console.log('\n');

/*
 * Q5.
 * There exists a staircase with N steps, and you can climb up either 1 or 2
 * steps at a time. Given N, write a function that returns the number of unique
 * ways you can climb the staircase. The order of the steps matters.
 * For example, if N is 4, then there are 5 unique ways:
 * 1, 1, 1, 1
 * 2, 1, 1
 * 1, 2, 1
 * 1, 1, 2
 * 2, 2
 * What if, instead of being able to climb 1 or 2 steps at a time, you could
 * climb any number from a set of positive integers X? For example, if X = {1,
 * 3, 5}, you could climb 1, 3, or 5 steps at a time.
 */
function countWays(n, X) {
    let dp = new Array(n + 1).fill(0);
    dp[0] = 1;

    for (let i = 1; i <= n; i++) {
        for (let j = 0; j < X.length; j++) {
            if (i >= X[j]) {
                dp[i] += dp[i - X[j]];
            }
        }
    }

    return dp[n];
}

console.log('========= Q5 =========');
const n = 7;
const X = [1, 3, 5];
console.log(
    `Number of unique ways to climb the staircase with ${n} steps using steps from set ${X}: ${countWays(
        n,
        X
    )}`
);
console.log('\n');

/*
 * Q6.
 * Given an integer k and a string s, find the length of the longest substring
 * that contains at most k distinct characters.
 * For example, given s = "abcba" and k = 2, the longest substring with k
 * distinct characters is "bcb".
 */
function longestSubstringWithDistinct(s, k) {
    if (!s || s.length === 0 || k <= 0) {
        return 0;
    }

    let maxLength = 0;
    let start = 0;
    let distinctCount = 0;
    let charCount = new Map();

    for (let end = 0; end < s.length; end++) {
        const c = s.charAt(end);
        charCount.set(c, charCount.get(c) + 1 || 1);

        if (charCount.get(c) == 1) {
            distinctCount++;
        }

        while (distinctCount > k) {
            const startChar = s.charAt(start);
            charCount.set(startChar, charCount.get(startChar) - 1);

            if (charCount.get(startChar) === 0) {
                distinctCount--;
            }
            start++;
        }

        maxLength = Math.max(maxLength, end - start + 1);
    }

    return maxLength;
}

console.log('========= Q6 =========');
const s = 'abcba';
const k = 2;
console.log(
    `Length of the longest substring with ${k} distinct characters: ${longestSubstringWithDistinct(
        s,
        k
    )}`
);
console.log('\n');

/*
 * Q7.
 * Suppose we represent our file system by a string in the following manner:
 * The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
 * " dir               "
 * "     subdir1       "
 * "     subdir2       "
 * "         file.ext  "
 * The directory dir contains an empty sub-directory subdir1 and a sub-directory
 * subdir2 containing a file file.ext.
 * The string
 * "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
 * represents:
 * " dir                       "
 * "     subdir1               "
 * "         file1.ext         "
 * "         subsubdir1        "
 * "     subdir2               "
 * "         subsubdir2        "
 * "             file2.ext     "
 * The directory dir contains two sub-directories subdir1 and subdir2. subdir1
 * contains a file file1.ext and an empty second-level sub-directory subsubdir1.
 * subdir2 contains a second-level sub-directory subsubdir2 containing a file
 * file2.ext.
 * We are interested in finding the longest (number of characters) absolute path
 * to a file within our file system. For example, in the second example above,
 * the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its
 * length is 32 (not including the double quotes).
 * Given a string representing the file system in the above format, return the
 * length of the longest absolute path to a file in the abstracted file system.
 * If there is no file in the system, return 0.
 * Note:
 * The name of a file contains at least a period and an extension.
 * The name of a directory or sub-directory will not contain a period.
 */
function lengthLongestPath(input) {
    if (!input || input.length === 0) {
        return 0;
    }

    let paths = input.split('\n');
    let stack = [];
    let maxLength = 0;

    for (const path of paths) {
        const level = getLevel(path);
        while (stack.length > level) {
            stack.pop();
        }

        const currLength =
            (stack.length === 0 ? 0 : stack[stack.length - 1]) +
            path.length -
            level +
            1;
        stack.push(currLength);

        if (isFile(path)) {
            maxLength = Math.max(maxLength, currLength - 1);
        }
    }

    return maxLength;
}

function getLevel(path) {
    let level = 0;
    let i = 0;
    while (i < path.length && path.charAt(i) === '\t') {
        level++;
        i++;
    }
    return level;
}

function isFile(path) {
    return path.indexOf('.') !== -1;
}

console.log('========= Q7 =========');
const input1 = 'dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext';
console.log(
    `Length of the longest absolute path: ${lengthLongestPath(input1)}`
);

const input2 =
    'dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext';
console.log(
    `Length of the longest absolute path: ${lengthLongestPath(input2)}`
);
console.log('\n');

/*
 * Q8.
 * Given an array of integers and a number k, where 1 <= k <= length of the
 * array, compute the maximum values of each subarray of length k.
 * For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10,
 * 7, 8, 8], since:
 * 10 = max(10, 5, 2)
 * 7 = max(5, 2, 7)
 * 8 = max(2, 7, 8)
 * 8 = max(7, 8, 7)
 * Do this in O(n) time and O(k) space. You can modify the input array in-place
 * and you do not need to store the results. You can simply print them out as
 * you compute them.
 */
function printMaxOfSubarrays(nums, k) {
    const deque = [];
    const n = nums.length;

    for (let i = 0; i < k; i++) {
        while (deque.length > 0 && nums[i] >= nums[deque[deque.length - 1]]) {
            deque.pop();
        }
        deque.push(i);
    }

    for (let i = k; i < n; i++) {
        console.log(nums[deque[0]]);

        // Remove elements outside the current window
        while (deque.length > 0 && deque[0] <= i - k) {
            deque.shift();
        }

        // Remove smaller elements from the deque
        while (deque.length > 0 && nums[i] >= nums[deque[deque.length - 1]]) {
            deque.pop();
        }

        deque.push(i);
    }

    // Print the maximum of the last subarray
    console.log(nums[deque[0]]);
}

console.log('========= Q8 =========');
const nums = [10, 5, 2, 7, 8, 7];
const kRange = 3;
printMaxOfSubarrays(nums, kRange);
console.log('\n');

/*
 * Q9.
 * Implement regular expression matching with the following special characters:
 * . (period) which matches any single character
 * * (asterisk) which matches zero or more of the preceding element
 * That is, implement a function that takes in a string and a valid regular
 * expression and returns whether or not the string matches the regular
 * expression.
 * For example, given the regular expression "ra." and the string "ray", your
 * function should return true. The same regular expression on the string
 * "raymond" should return false.
 * Given the regular expression ".*at" and the string "chat", your function
 * should return true. The same regular expression on the string "chats" should
 * return false.
 */
function isMatch(text, pattern) {
    if (pattern.length === 0) {
        return text.length === 0;
    }

    const firstMatch =
        text.length > 0 &&
        (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.');

    if (pattern.length >= 2 && pattern.charAt(1) == '*') {
        return (
            isMatch(text, pattern.substring(2)) ||
            (firstMatch && isMatch(text.substring(1), pattern))
        );
    } else {
        return firstMatch && isMatch(text.substring(1), pattern.substring(1));
    }
}

console.log('========= Q9 =========');
let text = 'ray';
let pattern = 'ra.';
console.log(isMatch(text, pattern)); // true

text = 'raymond';
pattern = 'ra.';
console.log(isMatch(text, pattern)); // false

text = 'chat';
pattern = '.*at';
console.log(isMatch(text, pattern)); // true

text = 'chats';
console.log(isMatch(text, pattern)); // false
console.log('\n');

/*
 * Q10.
 * Suppose you are given a table of currency exchange rates, represented as a 2D
 * array. Determine whether there is a possible arbitrage: that is, whether
 * there is some sequence of trades you can make, starting with some amount A of
 * any currency, so that you can end up with some amount greater than A of that
 * currency.
 * There are no transaction costs and you can trade fractional quantities.
 */
function hasArbitrage(rates) {
    const n = rates.length;
    let dist = new Array(n).fill(Number.MAX_SAFE_INTEGER);
    dist[0] = 0;

    // Relax edges repeatedly (Bellman-Ford algorithm)
    for (let i = 0; i < n - 1; i++) {
        for (let u = 0; u < n; u++) {
            for (let v = 0; v < n; v++) {
                if (dist[u] != Number.MAX_SAFE_INTEGER && rates[u][v] != 0) {
                    const newDist = dist[u] + Math.log(rates[u][v]);
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                    }
                }
            }
        }
    }

    // Check for negative cycles
    for (let u = 0; u < n; u++) {
        for (let v = 0; v < n; v++) {
            if (dist[u] != Number.MAX_SAFE_INTEGER && rates[u][v] != 0) {
                const newDist = dist[u] + Math.log(rates[u][v]);
                if (newDist < dist[v]) {
                    return true; // Negative cycle exists
                }
            }
        }
    }
    return false;
}

console.log('========= Q10 =========');
const rates = [
    [1.0, 0.05, 0.77],
    [1.18, 1.0, 0.91],
    [1.3, 1.1, 1.0],
];

console.log(`Arbitrage is${hasArbitrage(rates) ? '' : ' not'} possible`);
console.log('\n');

/*
 * Q11.
 * Given an array of strictly the characters 'R', 'G', and 'B', segregate the
 * values of the array so that all the Rs come first, the Gs come second, and
 * the Bs come last. You can only swap elements of the array.
 * Do this in linear time and in-place.
 * For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should
 * become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
 */
function segregateColors(colors) {
    let low = 0;
    let mid = 0;
    let high = colors.length - 1;

    while (mid <= high) {
        if (colors[mid] == 'R') {
            swap(colors, low, mid);
            low++;
            mid++;
        } else if (colors[mid] == 'G') {
            mid++;
        } else if (colors[mid] == 'B') {
            swap(colors, mid, high);
            high--;
        }
    }
}

function swap(colors, i, j) {
    const temp = colors[i];
    colors[i] = colors[j];
    colors[j] = temp;
}

console.log('========= Q11 =========');
const colors = ['G', 'B', 'R', 'R', 'B', 'R', 'G'];
segregateColors(colors);
console.log(colors);
console.log('\n');

/*
 * Q12.
 * You have an N by N board. Write a function that, given N, returns the number
 * of possible arrangements of the board where N queens can be placed on the
 * board without threatening each other, i.e. no two queens share the same row,
 * column, or diagonal.
 */
function countQueens(n) {
    let queens = new Array(n).fill(-1);
    return backtrack(queens, 0);
}

function backtrack(queens, row) {
    let count = 0;
    if (row === queens.length) {
        return 1;
    }

    for (let col = 0; col < queens.length; col++) {
        if (isSafe(queens, row, col)) {
            queens[row] = col;
            count += backtrack(queens, row + 1);
        }
    }

    return count;
}

function isSafe(queens, row, col) {
    for (let i = 0; i < row; i++) {
        if (
            queens[i] === col ||
            queens[i] - i === col - row ||
            queens[i] + i === col + row
        ) {
            return false;
        }
    }
    return true;
}

console.log('========= Q12 =========');
const N = 4;
console.log(`Number of possible arrangements: ${countQueens(N)}`);
console.log('\n');

/*
 * Q13.
 * Given an array of integers where every integer occurs three times except for
 * one integer, which only occurs once, find and return the non-duplicated
 * integer.
 * For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13],
 * return 19.
 * Do this in O(N) time and O(1) space.
 */
function findNonDuplicated(nums) {
    let ones = 0; // Count only bits that appear once
    let twos = 0; // Count only bits that appear twice

    for (const num of nums) {
        ones = (ones ^ num) & ~twos;
        twos = (twos ^ num) & ~ones;
    }

    return ones;
}

console.log('========= Q13 =========');
const numsToFindSingleAppearance1 = [6, 1, 3, 3, 3, 6, 6];
const numsToFindSingleAppearance2 = [13, 19, 13, 13];

console.log(
    `Non-duplicated number: ${findNonDuplicated(numsToFindSingleAppearance1)}`
);
console.log(
    `Non-duplicated number: ${findNonDuplicated(numsToFindSingleAppearance2)}`
);
console.log('\n');

/*
 * Q14.
 * Given a list of integers S and a target number k, write a function that
 * returns a subset of S that adds up to k. If such a subset cannot be made,
 * then return null.
 * Integers can appear more than once in the list. You may assume all numbers in
 * the list are positive.
 * For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1]
 * since it sums up to 24.
 */
function findSubset(nums, target) {
    let subset = [];
    const targetFound = backtrack(nums, target, 0, subset);
    return targetFound ? subset : null;
}

function backtrack(nums, target, index, subset) {
    if (target === 0) {
        return true;
    }

    for (let i = index; i < nums.length; i++) {
        if (nums[i] <= target) {
            subset.push(nums[i]);
            const targetFound = backtrack(
                nums,
                target - nums[i],
                i + 1,
                subset
            );

            if (targetFound) {
                return true;
            }
            subset.pop();
        }
    }
    return false;
}

console.log('========= Q14 =========');
const numsToFindSubset = [12, 1, 61, 5, 9, 2];
const target = 24;
console.log(`Subset: ${findSubset(numsToFindSubset, target)}`);
console.log('\n');

/*
 * Q15.
 * Given a string, find the longest palindromic contiguous substring. If there
 * are more than one with the maximum length, return any one.
 * For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The
 * longest palindromic substring of "bananas" is "anana".
 */
function longestPalindromicSubstring(s) {
    if (!s || s.length < 2) {
        return s;
    }

    let start = 0;
    let maxLength = 1;
    const n = s.length;
    let dp = new Array(n).fill(true).map(() => new Array(n).fill(true));

    for (let i = 0; i < n - 1; i++) {
        if (s[i] === s[i + 1]) {
            dp[i][i + 1] = true;
            start = i;
            maxLength = 2;
        }
    }

    for (let len = 3; len <= n; len++) {
        for (let i = 0; i <= n - len; i++) {
            const j = i + len - 1;
            if (s[i] === s[j] && dp[i + 1][j - 1]) {
                dp[i][j] = true;

                if (len > maxLength) {
                    start = i;
                    maxLength = len;
                }
            }
        }
    }
    return s.substring(start, start + maxLength);
}

console.log('========= Q15 =========');
const s1 = 'aabcdcb';
const s2 = 'bananas';

console.log(
    `Longest palindromic substring1: ${longestPalindromicSubstring(s1)}`
);
console.log(
    `Longest palindromic substring2: ${longestPalindromicSubstring(s2)}`
);
console.log('\n');

/*
 * Q16.
 * Implement an LRU (Least Recently Used) cache. It should be able to be
 * initialized with a cache size n, and contain the following methods:
 * set(key, value): sets key to value. If there are already n items in the cache
 * and we are adding a new item, then it should also remove the least recently
 * used item.
 * get(key): gets the value at key. If no such key exists, return null.
 * Each operation should run in O(1) time.
 */
class LRUCacheNode {
    constructor(key, value) {
        this.key = key;
        this.value = value;
        this.prev = null;
        this.next = null;
    }
}

class LRUCache {
    #_capacity;
    #_cache;
    #_head;
    #_tail;

    constructor(capacity) {
        this.#_capacity = capacity;
        this.#_cache = new Map();

        this.#_head = new LRUCacheNode(-1, -1);
        this.#_tail = new LRUCacheNode(-1, -1);
        this.#_head.next = this.#_tail;
        this.#_tail.prev = this.#_head;
    }

    get(key) {
        if (this.#_cache.has(key)) {
            const node = this.#_cache.get(key);
            this.#_removeNode(node);
            this.#_addNodeToHead(node);
            return node.value;
        }
        return -1;
    }

    set(key, value) {
        if (this.#_cache.has(key)) {
            const node = this.#_cache.get(key);
            node.value = value;
            this.#_removeNode(node);
            this.#_addNodeToHead(node);
        } else {
            const newNode = new LRUCacheNode(key, value);
            if (this.#_cache.size >= this.#_capacity) {
                const tailNode = this.#_tail.prev;
                this.#_removeNode(tailNode);
                this.#_cache.delete(this.#_tail.prev.key);
            }
            this.#_cache.set(key, newNode);
            this.#_addNodeToHead(newNode);
        }
    }

    #_removeNode(node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    #_addNodeToHead(node) {
        node.next = this.#_head.next;
        node.prev = this.#_head;
        this.#_head.next.prev = node;
        this.#_head.next = node;
    }
}

console.log('========= Q16 =========');
const cache = new LRUCache(3);
cache.set(1, 10);
cache.set(2, 20);
cache.set(3, 30);

console.log(`Get 1: ${cache.get(1)}`);
console.log(`Get 2: ${cache.get(2)}`);

cache.set(4, 40);

console.log(`Get 1: ${cache.get(1)}`);
console.log(`Get 3: ${cache.get(3)}`);
console.log(`Get 4: ${cache.get(4)}`);
console.log('\n');

/*
 * Q17.
 * Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with
 * digits. The objective is to fill the grid with the constraint that every row,
 * column, and box (3 by 3 subgrid) must contain all of the digits from 1 to 9.
 * Implement an efficient sudoku solver.
 */
function solveSudoku(grid) {
    for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
            if (grid[row][col] === 0) {
                for (let digit = 1; digit <= 9; digit++) {
                    if (isValid(grid, row, col, digit)) {
                        grid[row][col] = digit;
                        if (solveSudoku(grid)) {
                            return true;
                        } else {
                            grid[row][col] = 0;
                        }
                    }
                }
                return false;
            }
        }
    }
    return true;
}

function isValid(grid, row, col, digit) {
    for (let i = 0; i < 9; i++) {
        if (grid[row][i] === digit || grid[i][col] === digit) {
            return false;
        }
    }

    const subgridStartRow = Math.floor(row / 3) * 3;
    const subgridStartCol = Math.floor(col / 3) * 3;

    for (let i = subgridStartRow; i < subgridStartRow + 3; i++) {
        for (let j = subgridStartCol; j < subgridStartCol + 3; j++) {
            if (grid[i][j] === digit) {
                return false;
            }
        }
    }
    return true;
}

console.log('========= Q17 =========');
const grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

if (solveSudoku(grid)) {
    console.log(grid);
} else {
    console.log('No solution exists');
}
console.log('\n');

/*
 * Q18.
 * Implement a file syncing algorithm for two computers over a low-bandwidth
 * network. What if we know the files in the two computers are mostly the same?
 */
console.log('========= Q18 =========');
/*
 * 1. Establish a TCP connection between the two computers.
 * 2. Identify the files that need to be synced. Maintain list of files on each
 * computer and compare the differences.
 * 3. Use a differential syncing algorithm to transfer only different parts.
 * - Implement a mechanism to compare files to identify the differences such as
 * file hashing or timestamp comparison.
 * - When a file is modified, calculate the difference between old and new
 * version. This can be done by comparing contents of the files or using delta
 * encoding techniques.
 * 4. Transfer the differences over the low-bandwidth network. This can be done
 * by sending only the modified parts of the files or using compression
 * techniques to reduce data size.
 * 5. On the receiving computer, apply the received differences to update the
 * corresponding files. This can involve patching or merging the changes into
 * the existing files.
 * 6. Repeat synching process periodically or whenever changes are detected on
 * either computer.
 * 7. Implement error handling and recovery mechanisms to handle network
 * failures, file conflicts, etc.
 * 8. Monitor the syncing process and provide feedback to the users about the
 * progress and status of the synchronization.
 */
console.log('\n');

/*
 * Q19.
 * A knight's tour is a sequence of moves by a knight on a chessboard such that
 * all squares are visited once.
 * Given N, write a function to return the number of knight's tours on an N by N
 * chessboard.
 */
class KnightsTour {
    #_ROWS_MOVES = [2, 1, -1, -2, -2, -1, 1, 2];
    #_COLS_MOVES = [1, 2, 2, 1, -1, -2, -2, -1];

    countKnightTours(n) {
        let board = new Array(n).fill(0).map(() => new Array(n).fill(0));
        let count = 0;

        for (let row = 0; row < n; row++) {
            for (let col = 0; col < n; col++) {
                count += this.#_findTours(board, row, col, 1);
            }
        }
        return count;
    }

    #_findTours(board, row, col, moveCount) {
        const n = board.length;

        if (moveCount === n * n) {
            return 1;
        }

        let count = 0;
        board[row][col] = moveCount;

        for (let i = 0; i < 8; i++) {
            const nextRow = row + this.#_ROWS_MOVES[i];
            const nextCol = col + this.#_COLS_MOVES[i];

            if (this.#_isValidMove(board, nextRow, nextCol)) {
                count += this.#_findTours(
                    board,
                    nextRow,
                    nextCol,
                    moveCount + 1
                );
            }
        }

        board[row][col] = 0;
        return count;
    }

    #_isValidMove(board, row, col) {
        const n = board.length;
        return (
            row >= 0 && row < n && col >= 0 && col < n && board[row][col] === 0
        );
    }
}

console.log('========= Q19 =========');
const knightsTour = new KnightsTour();
console.log(knightsTour.countKnightTours(5));
console.log('\n');

/*
 * Q20.
 * Implement an LFU (Least Frequently Used) cache. It should be able to be
 * initialized with a cache size n, and contain the following methods:
 * set(key, value): sets key to value. If there are already n items in the cache
 * and we are adding a new item, then it should also remove the least frequently
 * used item. If there is a tie, then the least recently used key should be
 * removed.
 * get(key): gets the value at key. If no such key exists, return null.
 * Each operation should run in O(1) time.
 */
class LFUCache {
    #_capacity;
    #_minFrequency;
    #_keyToValue;
    #_keyToFrequency;
    #_frequencyToKeys;

    constructor(capacity) {
        this.#_capacity = capacity;
        this.#_minFrequency = 0;
        this.#_keyToValue = new Map();
        this.#_keyToFrequency = new Map();
        this.#_frequencyToKeys = new Map();
    }

    get(key) {
        if (!this.#_keyToValue.has(key)) {
            return -1;
        }

        const frequency = this.#_keyToFrequency.get(key);
        this.#_keyToFrequency.set(key, frequency + 1);
        this.#_frequencyToKeys.get(frequency).delete(key);

        if (this.#_frequencyToKeys.get(frequency).size === 0) {
            this.#_frequencyToKeys.delete(frequency);

            if (frequency === this.#_minFrequency) {
                this.#_minFrequency++;
            }
        }

        this.#_frequencyToKeys.set(
            frequency + 1,
            this.#_frequencyToKeys.get(frequency + 1)
                ? this.#_frequencyToKeys.get(frequency + 1).add(key)
                : new Set([key])
        );

        return this.#_keyToValue.get(key);
    }

    set(key, value) {
        if (this.#_capacity <= 0) {
            return;
        }

        if (this.#_keyToValue.has(key)) {
            this.#_keyToValue.set(key, value);
            this.get(key);
            return;
        }

        if (this.#_keyToValue.size >= this.#_capacity) {
            const evictKey = this.#_frequencyToKeys
                .get(this.#_minFrequency)
                .values()
                .next().value;
            this.#_frequencyToKeys.get(this.#_minFrequency).delete(evictKey);
            this.#_keyToValue.delete(evictKey);
            this.#_keyToFrequency.delete(evictKey);
        }

        this.#_keyToValue.set(key, value);
        this.#_keyToFrequency.set(key, 1);
        this.#_frequencyToKeys.set(
            1,
            this.#_frequencyToKeys.get(1)
                ? this.#_frequencyToKeys.get(1).add(key)
                : new Set([key])
        );
        this.#_minFrequency = 1;
    }
}

console.log('========= Q20 =========');
const lfuCache = new LFUCache(2);

lfuCache.set(1, 10);
lfuCache.set(2, 20);
console.log(lfuCache.get(1)); // Output: 10

lfuCache.set(3, 30);
console.log(lfuCache.get(2)); // Output: -1
console.log(lfuCache.get(3)); // Output: 30

lfuCache.set(4, 40);
console.log(lfuCache.get(1)); // Output: -1
console.log(lfuCache.get(3)); // Output: 30
console.log(lfuCache.get(4)); // Output: 40
console.log('\n');

/*
 * Q21.
 * In a directed graph, each node is assigned an uppercase letter. We define a
 * path's value as the number of most frequently-occurring letter along that
 * path. For example, if a path in the graph goes through "ABACA", the value of
 * the path is 3, since there are 3 occurrences of 'A' on the path.
 * Given a graph with n nodes and m directed edges, return the largest value
 * path of the graph. If the largest value is infinite, then return null.
 * The graph is represented with a string and an edge list. The i-th character
 * represents the uppercase letter of the i-th node. Each tuple in the edge list
 * (i, j) means there is a directed edge from the i-th node to the j-th node.
 * Self-edges are possible, as well as multi-edges.
 * For example, the following input graph:
 * ABACA
 * [(0, 1),
 * (0, 2),
 * (2, 3),
 * (3, 4)]
 * Would have maximum value 3 using the path of vertices [0, 2, 3, 4], (A, A, C,
 * A).
 * The following input graph:
 * A
 * [(0, 0)]
 * Should return null, since we have an infinite loop.
 */
class DirectedGraph {
    #_adjacencyList;
    #_nodeValues;

    constructor(nodes, edges) {
        this.#_adjacencyList = new Map();
        this.#_nodeValues = new Array(nodes.length);

        for (let i = 0; i < nodes.length; i++) {
            this.#_adjacencyList.set(i, new Set());
            this.#_nodeValues[i] = nodes[i];
        }

        for (const [from, to] of edges) {
            this.#_adjacencyList.get(from).add(to);
        }
    }

    findLargestValuePath() {
        const numNodes = this.#_nodeValues.length;
        let visited = new Array(numNodes).fill(false);
        let freqMap = new Map();
        let maxValue = 0;

        for (let i = 0; i < numNodes; i++) {
            if (!visited[i]) {
                const pathValue = this.dfs(i, visited, freqMap);

                if (!pathValue) {
                    return null;
                }
                maxValue = Math.max(maxValue, pathValue);
            }
        }

        return maxValue;
    }

    dfs(node, visited, freqMap) {
        visited[node] = true;
        const currNode = this.#_nodeValues[node];
        freqMap.set(currNode, freqMap.get(currNode) + 1 || 1);
        const neighbors = this.#_adjacencyList.get(node) || new Set();
        let maxValue = freqMap.get(currNode);

        for (const neighbor of neighbors) {
            if (visited[neighbor]) {
                return null;
            }

            const pathValue = this.dfs(neighbor, visited, freqMap);
            if (!pathValue) {
                return null;
            }
            maxValue = Math.max(maxValue, pathValue);
        }

        visited[node] = false;
        freqMap.set(currNode, freqMap.get(currNode) - 1);
        if (freqMap.get(currNode) === 0) {
            freqMap.delete(currNode);
        }

        return maxValue;
    }
}

console.log('========= Q21 =========');
const nodes1 = 'ABACA';
const edges1 = [
    [0, 1],
    [0, 2],
    [2, 3],
    [3, 4],
];
const graph1 = new DirectedGraph(nodes1, edges1);
console.log(graph1.findLargestValuePath()); // Output: 3

const nodes2 = 'A';
const edges2 = [[0, 0]];
const graph2 = new DirectedGraph(nodes2, edges2);
console.log(graph2.findLargestValuePath()); // Output: null
console.log('\n');

/*
 * Q22.
 * Given an array of numbers, find the length of the longest increasing
 * subsequence in the array. The subsequence does not necessarily have to be
 * contiguous.
 * For example, given the array [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11,
 * 7, 15], the longest increasing subsequence has length 6: it is 0, 2, 6, 9,
 * 11, 15.
 */
function lengthOfLIS(nums) {
    const n = nums.length;
    if (n === 0) {
        return 0;
    }

    let dp = new Array(n).fill(0);
    dp[0] = 1;
    let maxLength = 1;

    for (let i = 1; i < n; i++) {
        let maxVal = 0;
        for (let j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                maxVal = Math.max(maxVal, dp[j]);
            }
        }
        dp[i] = maxVal + 1;
        maxLength = Math.max(maxLength, dp[i]);
    }
    return maxLength;
}

console.log('========= Q22 =========');
const numsToFindLIS = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
console.log(lengthOfLIS(numsToFindLIS)); // Output: 6
console.log('\n');

/*
 * Q23.
 * A rule looks like this:
 * A NE B
 * This means this means point A is located northeast of point B.
 * A SW C
 * means that point A is southwest of C.
 * Given a list of rules, check if the sum of the rules validate. For example:
 * A N B
 * B NE C
 * C N A
 * does not validate, since A cannot be both north and south of C.
 * A NW B
 * A N B
 * is considered valid.
 */
function validateRules(rules) {
    let relationships = new Map();

    for (const rule of rules) {
        const parts = rule.split(' ');
        const point1 = parts[0];
        const direction = parts[1];
        const point2 = parts[2];

        if (!validateRule(relationships, point1, direction, point2)) {
            return false;
        }
    }

    return true;
}

function validateRule(relationships, point1, direction, point2) {
    let existingDirection1 = relationships.get(point1) || new Set();
    let existingDirection2 = relationships.get(point2) || new Set();

    if (
        existingDirection1.has(directionOpposite(direction)) ||
        existingDirection2.has(direction)
    ) {
        return false;
    }

    existingDirection1.add(direction);
    existingDirection2.add(directionOpposite(direction));

    relationships.set(point1, existingDirection1);
    relationships.set(point2, existingDirection2);

    return true;
}

function directionOpposite(direction) {
    switch (direction) {
        case 'N':
            return 'S';
        case 'S':
            return 'N';
        case 'E':
            return 'W';
        case 'W':
            return 'E';
        case 'NE':
            return 'SW';
        case 'SW':
            return 'NE';
        case 'NW':
            return 'SE';
        case 'SE':
            return 'NW';
        default:
            return '';
    }
}

function addRelationship(relationships, point1, direction, point2) {
    if (!relationships.has(point1)) {
        relationships.set(point1, new Set());
    }
    relationships.get(point1).add(direction);

    if (!relationships.has(point2)) {
        relationships.set(point2, new Set());
    }
    relationships.get(point2).add(directionOpposite(direction));
}

console.log('========= Q23 =========');
const rules1 = ['A N B', 'C SE B', 'C N A'];
console.log(validateRules(rules1)); // Output: false

const rules2 = ['A NW B', 'A N B'];
console.log(validateRules(rules2)); // Output: true
console.log('\n');

/*
 * Q24.
 * We're given a hashmap associating each courseId key with a list of courseIds
 * values, which represents that the prerequisites of courseId are courseIds.
 * Return a sorted ordering of courses such that we can finish all courses.
 * Return null if there is no such ordering.
 * For example, given {'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'],
 * 'CSC100': []}, should return ['CSC100', 'CSC200', 'CSCS300'].
 */
function findCourseOrder(prerequisites) {
    let graph = new buildGraph(prerequisites);
    let visited = new Set();
    let visiting = new Set();
    let courseOrder = [];

    for (const course of graph.keys()) {
        if (
            !visited.has(course) &&
            !dfs(course, graph, visited, visiting, courseOrder)
        ) {
            return null;
        }
    }

    return courseOrder;
}

function buildGraph(prerequisites) {
    let graph = new Map();

    for (const course of prerequisites.keys()) {
        graph.set(course, []);
    }

    for (const course of prerequisites.keys()) {
        for (const prerequisite of prerequisites.get(course)) {
            graph.get(course).push(prerequisite);
        }
    }

    return graph;
}

function dfs(course, graph, visited, visiting, courseOrder) {
    visiting.add(course);

    for (const prerequisite of graph.get(course)) {
        if (visiting.has(prerequisite)) {
            return false;
        }

        if (
            !visited.has(prerequisite) &&
            !dfs(prerequisite, graph, visited, visiting, courseOrder)
        ) {
            return false;
        }
    }

    visiting.delete(course);
    visited.add(course);
    courseOrder.push(course);
    return true;
}

console.log('========= Q24 =========');
const prerequisites = new Map();
prerequisites.set('CSC300', ['CSC100', 'CSC200']);
prerequisites.set('CSC200', ['CSC100']);
prerequisites.set('CSC100', []);

const courseOrder = findCourseOrder(prerequisites);
if (!courseOrder) {
    console.log('No valid course ordering exists.');
} else {
    console.log(`Course ordering: ${courseOrder}`);
}
console.log('\n');

/*
 * Q25.
 * Given a tree, find the largest tree/subtree that is a BST.
 * Given a tree, return the size of the largest tree/subtree that is a BST.
 */
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

class Result {
    constructor(size, min, max, isBST) {
        this.size = size;
        this.min = min;
        this.max = max;
        this.isBST = isBST;
    }
}

function largestBST(root) {
    return largestBSTHelper(root).size;
}

function largestBSTHelper(root) {
    if (!root) {
        return new Result(
            0,
            Number.MAX_SAFE_INTEGER,
            Number.MIN_SAFE_INTEGER,
            true
        );
    }

    const left = largestBSTHelper(root.left);
    const right = largestBSTHelper(root.right);

    if (
        !left.isBST ||
        !right.isBST ||
        root.val < left.max ||
        root.val > right.min
    ) {
        return new Result(Math.max(left.size, right.size), 0, 0, false);
    }

    const size = left.size + right.size + 1;
    const min = root.left ? left.min : root.val;
    const max = root.right ? right.max : root.val;

    return new Result(size, min, max, true);
}

console.log('========= Q25 =========');
const root = new TreeNode(10);
root.left = new TreeNode(5);
root.right = new TreeNode(15);
root.left.left = new TreeNode(1);
root.left.right = new TreeNode(8);
root.right.right = new TreeNode(7);

console.log(`Size of the largest BST: ${largestBST(root)}`); // Output: 3
console.log('\n');

/*
 * Q26.
 * Given a number represented by a list of digits, find the next greater
 * permutation of a number, in terms of lexicographic ordering. If there is not
 * greater permutation possible, return the permutation with the lowest
 * value/ordering.
 * For example, the list [1,2,3] should return [1,3,2]. The list [1,3,2] should
 * return [2,1,3]. The list [3,2,1] should return [1,2,3].
 * Can you perform the operation without allocating extra memory (disregarding
 * the input memory)?
 */
function nextPermutation(nums) {
    const n = nums.length;
    let i = n - 2;

    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    if (i >= 0) {
        let j = n - 1;
        while (j >= 0 && nums[j] <= nums[i]) {
            j--;
        }
        swap(nums, i, j);
    }

    reverse(nums, i + 1, n - 1);
}

function swap(nums, i, j) {
    const temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}

function reverse(nums, start, end) {
    while (start < end) {
        swap(nums, start, end);
        start++;
        end--;
    }
}

console.log('========= Q26 =========');
const numsToFindNextPermutation1 = [1, 2, 3];
nextPermutation(numsToFindNextPermutation1);
console.log(`Next permutation: ${numsToFindNextPermutation1}`); // Output: [1, 3, 2]

const numsToFindNextPermutation2 = [1, 3, 2];
nextPermutation(numsToFindNextPermutation2);
console.log(`Next permutation: ${numsToFindNextPermutation2}`); // Output: [2, 1, 3]

const numsToFindNextPermutation3 = [3, 2, 1];
nextPermutation(numsToFindNextPermutation3);
console.log(`Next permutation: ${numsToFindNextPermutation3}`); // Output: [1, 2, 3]
console.log('\n');

/*
 * Q27.
 * Given a word W and a string S, find all starting indices in S which are
 * anagrams of W.
 * For example, given that W is "ab", and S is "abxaba", return 0, 3, and 4.
 */
function findAnagramIndices(s, w) {
    let result = [];

    if (s.length === 0 || w.length === 0 || s.length < w.length) {
        return result;
    }

    let targetFreqMap = new Map();
    let windowFreqMap = new Map();

    for (const ch of w) {
        targetFreqMap.set(ch, (targetFreqMap.get(ch) || 0) + 1);
    }

    const windowSize = w.length;
    for (let i = 0; i < windowSize; i++) {
        const ch = s[i];
        windowFreqMap.set(ch, (windowFreqMap.get(ch) || 0) + 1);
    }

    if (isAnagram(targetFreqMap, windowFreqMap)) {
        result.push(0);
    }

    for (let i = windowSize; i < s.length; i++) {
        const incoming = s[i];
        const outgoing = s[i - windowSize];

        windowFreqMap.set(incoming, (windowFreqMap.get(incoming) || 0) + 1);
        windowFreqMap.set(outgoing, windowFreqMap.get(outgoing) - 1);

        if (windowFreqMap.get(outgoing) === 0) {
            windowFreqMap.delete(outgoing);
        }

        if (isAnagram(targetFreqMap, windowFreqMap)) {
            result.push(i - windowSize + 1);
        }
    }

    return result;
}

function isAnagram(targetFreqMap, windowFreqMap) {
    return (
        targetFreqMap.size === windowFreqMap.size &&
        [...targetFreqMap].every(([ch, freq]) => windowFreqMap.get(ch) === freq)
    );
}

console.log('========= Q27 =========');
const S = 'abxaba';
const W = 'ab';
console.log(`Anagram indices: ${findAnagramIndices(S, W)}`); // Output: [0, 3, 4]

/*
 * Q28.
 * Given a binary tree, find the lowest common ancestor (LCA) of two given nodes
 * in the tree. Assume that each node in the tree also has a pointer to its
 * parent.
 * https://en.wikipedia.org/wiki/Lowest_common_ancestor
 * According to the definition of LCA on Wikipedia: “The lowest common ancestor
 * is defined between two nodes v and w as the lowest node in T that has both v
 * and w as descendants (where we allow a node to be a descendant of itself).”
 */
class TreeNodeWithParent {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
        this.parent = null;
    }
}

function findLowestCommonAncestor(root, p, q) {
    if (!root || !p || !q) {
        return null;
    }

    const pathToP = getPathToRoot(p);
    const pathToQ = getPathToRoot(q);

    let i = 0;
    while (i < pathToP.length && i < pathToQ.length) {
        if (pathToP[i] !== pathToQ[i]) {
            break;
        }
        i++;
    }

    if (i > 0) {
        return pathToP[i - 1];
    }

    return null;
}

function getPathToRoot(node) {
    let path = [];

    while (node) {
        path.push(node);
        node = node.parent;
    }

    return path.reverse();
}

console.log('========= Q28 =========');
const treeWithParent = new TreeNodeWithParent(3);
treeWithParent.left = new TreeNodeWithParent(5);
treeWithParent.right = new TreeNodeWithParent(1);

treeWithParent.left.parent = treeWithParent;
treeWithParent.right.parent = treeWithParent;

treeWithParent.left.left = new TreeNodeWithParent(6);
treeWithParent.left.right = new TreeNodeWithParent(2);

treeWithParent.left.left.parent = treeWithParent.left;
treeWithParent.left.right.parent = treeWithParent.left;

treeWithParent.left.right.left = new TreeNodeWithParent(7);
treeWithParent.left.right.right = new TreeNodeWithParent(4);

treeWithParent.left.right.left.parent = treeWithParent.left.right;
treeWithParent.left.right.right.parent = treeWithParent.left.right;

treeWithParent.right.left = new TreeNodeWithParent(0);
treeWithParent.right.right = new TreeNodeWithParent(8);

treeWithParent.right.left.parent = treeWithParent.right;
treeWithParent.right.right.parent = treeWithParent.right;

const p = treeWithParent.left;
const q = treeWithParent.right;

const lca = findLowestCommonAncestor(treeWithParent, p, q);
if (lca) {
    console.log(`Lowest Common Ancestor: ${lca.val}`);
} else {
    console.log(`Lowest Common Ancestor not found`);
}
console.log('\n');

/*
 * Q29.
 * Given a string and a set of delimiters, reverse the words in the string while
 * maintaining the relative order of the delimiters. For example, given
 * "hello/world:here", return "here/world:hello"
 * Follow-up: Does your solution work for the following cases:
 * "hello/world:here/", "hello//world:here"
 */
function reverseWordsWithDelimiters(str, delimiters) {
    let words = str.split(
        new RegExp(`[${getDelimitersAsString(delimiters)}]+`)
    );
    let separators = str.split(
        new RegExp(`[^${getDelimitersAsString(delimiters)}]+`)
    );
    separators = separators.filter((separator) => separator !== '');

    reverseArray(words);

    let reversed = '';
    let i = 0;
    let j = 0;
    while (i < words.length || j < separators.length) {
        if (i < words.length) {
            reversed += words[i++];
        }
        if (j < separators.length) {
            reversed += separators[j++];
        }
    }

    return reversed;
}

function getDelimitersAsString(delimiters) {
    let str = '';
    for (const delimiter of delimiters) {
        str += delimiter;
    }
    return str;
}

function reverseArray(arr) {
    let start = 0;
    let end = arr.length - 1;

    while (start < end) {
        const temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

console.log('========= Q29 =========');
const str1 = 'hello/world:here';
const str2 = 'hello/world:here/';
const str3 = 'hello//world:here';
const delimiters = new Set(['/', ':']);

let reversed = reverseWordsWithDelimiters(str1, delimiters);
console.log(`Reversed string: ${reversed}`);
reversed = reverseWordsWithDelimiters(str2, delimiters);
console.log(`Reversed string: ${reversed}`);
reversed = reverseWordsWithDelimiters(str3, delimiters);
console.log(`Reversed string: ${reversed}`);
console.log('\n');

/*
 * Q30.
 * Given two non-empty binary trees s and t, check whether tree t has exactly
 * the same structure and node values with a subtree of s. A subtree of s is a
 * tree consists of a node in s and all of this node's descendants. The tree s
 * could also be considered as a subtree of itself.
 */
function isSubtree(s, t) {
    if (!s) {
        return false;
    }

    if (isSameTree(s, t)) {
        return true;
    }

    return isSubtree(s.left, t) || isSubtree(s.right, t);
}

function isSameTree(s, t) {
    if (!s && !t) {
        return true;
    }

    if (!s || !t) {
        return false;
    }

    return (
        s.val === t.val &&
        isSameTree(s.left, t.left) &&
        isSameTree(s.right, t.right)
    );
}

console.log('========= Q30 =========');
const binaryTreeS = new TreeNode(3);
binaryTreeS.left = new TreeNode(4);
binaryTreeS.right = new TreeNode(5);
binaryTreeS.left.left = new TreeNode(1);
binaryTreeS.left.right = new TreeNode(2);
binaryTreeS.left.right.left = new TreeNode(0);

const binaryTreeT = new TreeNode(4);
binaryTreeT.left = new TreeNode(1);
binaryTreeT.right = new TreeNode(2);

console.log(`Is t a subtree of s? ${isSubtree(binaryTreeS, binaryTreeT)}`);
console.log('\n');

/*
 * Q31.
 * Given a string which we can delete at most k, return whether you can make a
 * palindrome.
 * For example, given 'waterrfetawx' and a k of 2, you could delete f and x to
 * get 'waterretaw'.
 */
function canMakePalindrome(s, k) {
    const n = s.length;
    let dp = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let len = 2; len <= n; len++) {
        for (let i = 0; i < n - len; i++) {
            let j = i + len - 1;

            if (s[i] === s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
            }
        }
    }

    return dp[0][n - 1] <= k;
}

console.log('========= Q31 =========');
const palindromeCandidate = 'waterrfetawx';
const numOfRemoval = 2;
console.log(
    `Can make palindrome: ${canMakePalindrome(
        palindromeCandidate,
        numOfRemoval
    )}`
);
console.log('\n');

/*
 * Q32.
 * Given a string, return whether it represents a number. Here are the different
 * kinds of numbers:
 * "10", a positive integer
 * "-10", a negative integer
 * "10.1", a positive real number
 * "-10.1", a negative real number
 * "1e5", a number in scientific notation
 * And here are examples of non-numbers:
 * "a"
 * "x 1"
 * "a -2"
 * "-"
 */
function isNumber(s) {
    s = s.trim();
    const pattern = /^[-+]?(?:\d+\.?|\.\d+)\d*(?:e[-+]?\d+)?$/;
    return pattern.test(s);
}

console.log('========= Q32 =========');
const inputs = ['10', '-10', '10.1', '-10.1', '1e5', 'a', 'x 1', 'a -2', '-'];
for (const input of inputs) {
    console.log(`${input} is a number: ${isNumber(input)}`);
}
console.log('\n');

/*
 * Q33.
 * Find the minimum number of coins required to make n cents.
 * You can use standard American denominations, that is, 1¢, 5¢, 10¢, and 25¢.
 * For example, given n = 16, return 3 since we can make it with a 10¢, a 5¢,
 * and a 1¢.
 */
function coinChange(n) {
    let coins = [25, 10, 5, 1];

    let dp = new Array(n + 1).fill(n + 1);
    dp[0] = 0;

    for (let i = 1; i <= n; i++) {
        for (const coin of coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[n] > n ? -1 : dp[n];
}

console.log('========= Q33 =========');
const sum = 16;
console.log(
    `Minimum number of coins required to make ${sum} cents: ${coinChange(sum)}`
);
console.log('\n');

/*
 * Q34.
 * Implement 3 stacks using a single list:
 * " class Stack:                              "
 * "     def __init__(self):                   "
 * "         self.list = []                    "
 *
 * "     def pop(self, stack_number):          "
 * "         pass                              "
 *
 * "     def push(self, item, stack_number):   "
 * "         pass                              "
 */
class MultiStack {
    #_list;
    #_tops;
    #_stackSize;
    #_numStacks;

    constructor(stackSize, numStacks) {
        this.#_stackSize = stackSize;
        this.#_numStacks = numStacks;
        this.#_list = new Array(stackSize * numStacks);
        this.#_tops = new Array(numStacks).fill(-1);
    }

    push(item, stackNumber) {
        if (this.isFull(stackNumber)) {
            console.log(
                `Stack ${stackNumber} is full. Cannot push item: ${item}`
            );
            return;
        }

        let index = this.getTopIndex(stackNumber);
        index++;
        this.#_list[this.#_stackSize * stackNumber + index] = item;
        this.#_tops[stackNumber] = index;
    }

    pop(stackNumber) {
        if (this.isEmpty(stackNumber)) {
            console.log(`Stack ${stackNumber} is empty. Cannot pop item.`);
            return -1;
        }

        let index = this.getTopIndex(stackNumber);
        const item = this.#_list[this.#_stackSize * stackNumber + index];
        this.#_tops[stackNumber] = index - 1;
        return item;
    }

    isEmpty(stackNumber) {
        return this.#_tops[stackNumber] === -1;
    }

    isFull(stackNumber) {
        return (
            this.#_tops[stackNumber] ===
            this.#_stackSize * (stackNumber + 1) - 1
        );
    }

    getTopIndex(stackNumber) {
        return this.#_tops[stackNumber];
    }
}

console.log('========= Q34 =========');
const stack = new MultiStack(10, 3);
stack.push(1, 0);
stack.push(2, 0);
stack.push(3, 1);
stack.push(4, 1);
stack.push(5, 2);
stack.push(6, 2);

console.log(`Pop from stack 0: ${stack.pop(0)}`);
console.log(`Pop from stack 1: ${stack.pop(1)}`);
console.log(`Pop from stack 2: ${stack.pop(2)}`);
console.log('\n');

/*
 * Q35.
 * You're given a string consisting solely of (, ), and *. * can represent
 * either a (, ), or an empty string. Determine whether the parentheses are
 * balanced.
 * For example, (()* and (*) are balanced. )*( is not balanced.
 */
function isBalanced(str) {
    let minOpen = 0;
    let maxOpen = 0;

    for (const c of str) {
        if (c == '(') {
            minOpen++;
            maxOpen++;
        } else if (c == ')') {
            minOpen = Math.max(minOpen - 1, 0);
            maxOpen--;
        } else {
            minOpen = Math.max(minOpen - 1, 0);
            maxOpen++;
        }

        if (maxOpen < 0) {
            return false; // More closing parentheses encountered than open parentheses
        }
    }

    return minOpen == 0;
}

console.log('========= Q35 =========');
const string1 = '(()*';
const string2 = '(*)';
const string3 = ')*(';

console.log(`Is ${string1} balanced: ${isBalanced(string1)}`);
console.log(`Is ${string2} balanced: ${isBalanced(string2)}`);
console.log(`Is ${string3} balanced: ${isBalanced(string3)}`);
console.log('\n');

/*
 * Q36.
 * Given a list, sort it using this method: reverse(lst, i, j), which reverses
 * lst from i to j.
 */
function sort(lst) {
    const n = lst.length;
    for (let i = 0; i < n; i++) {
        let minIndex = i;
        for (let j = i + 1; j < n; j++) {
            if (lst[j] < lst[minIndex]) {
                minIndex = j;
            }
        }
        reverse(lst, i, minIndex);
    }
}

function reverse(lst, i, j) {
    while (i < j) {
        const temp = lst[i];
        lst[i] = lst[j];
        lst[j] = temp;
        i++;
        j--;
    }
}

console.log('========= Q36 =========');
const lst = [9, 2, 5, 1, 7];
console.log(`Original list: ${lst}`);
sort(lst);
console.log(`Sorted list: ${lst}`);
console.log('\n');

/*
 * Q37.
 * Given a list of numbers L, implement a method sum(i, j) which returns the sum
 * from the sublist L[i:j] (including i, excluding j).
 * For example, given L = [1, 2, 3, 4, 5], sum(1, 3) should return sum([2, 3]),
 * which is 5.
 * You can assume that you can do some pre-processing. sum() should be optimized
 * over the pre-processing step.
 */
class SublistSum {
    #_prefixSums;

    constructor(lst) {
        this.#_prefixSums = [];
        let sum = 0;
        this.#_prefixSums.push(0);
        for (const num of lst) {
            sum += num;
            this.#_prefixSums.push(sum);
        }
    }

    sum(i, j) {
        if (i < 0 || j > this.#_prefixSums.length - 1 || i > j) {
            throw new Error('Invalid sublist range');
        }
        return this.#_prefixSums[j] - this.#_prefixSums[i];
    }
}

console.log('========= Q37 =========');
const L = [1, 2, 3, 4, 5];
const sublistSum = new SublistSum(L);

console.log(`Sum of sublist [1:3]: ${sublistSum.sum(1, 3)}`);
console.log(`Sum of sublist [2:5]: ${sublistSum.sum(2, 5)}`);
console.log(`Sum of sublist [0:5]: ${sublistSum.sum(0, 5)}`);
console.log('\n');

/*
 * Q38.
 * Given a list of points, a central point, and an integer k, find the nearest k
 * points from the central point.
 * For example, given the list of points [(0, 0), (5, 4), (3, 1)], the central
 * point (1, 2), and k = 2, return [(0, 0), (3, 1)].
 */
class Point {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

function findNearestPoints(points, centralPoint, k) {
    let pq = [];

    for (const point of points) {
        pq.push(point);
        pq.sort(
            (p1, p2) =>
                calculateDistance(p2, centralPoint) -
                calculateDistance(p1, centralPoint)
        );

        if (pq.length > k) {
            pq.shift();
        }
    }

    return pq;
}

function calculateDistance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

console.log('========= Q38 =========');
const points = [new Point(0, 0), new Point(5, 4), new Point(3, 1)];
const centralPoint = new Point(1, 2);
const numOfPoints = 2;
const nearestPoints = findNearestPoints(points, centralPoint, numOfPoints);
console.log(
    `Nearest ${numOfPoints} points to (${centralPoint.x}: ${nearestPoints.y}): `
);
for (const point of nearestPoints) {
    console.log(`(${point.x}, ${point.y})`);
}
console.log('\n');

/*
 * Q39.
 * Find an efficient algorithm to find the smallest distance (measured in number
 * of words) between any two given words in a string.
 * For example, given words "hello", and "world" and a text content of
 * "dog cat hello cat dog dog hello cat world", return 1 because there's only
 * one word "cat" in between the two words.
 */
function findSmallestDistance(text, word1, word2) {
    const words = text.split(' ');
    let minDistance = Number.MAX_SAFE_INTEGER;
    let prevIndex = -1;

    for (let i = 0; i < words.length; i++) {
        if (words[i] === word1) {
            if (prevIndex !== -1 && i - prevIndex < minDistance) {
                minDistance = i - prevIndex;
            }
            prevIndex = i;
        } else if (words[i] === word2) {
            if (prevIndex !== -1 && i - prevIndex < minDistance) {
                minDistance = i - prevIndex;
            }
            prevIndex = i;
        }
    }
    return minDistance - 1;
}

console.log('========= Q39 =========');
const textToFindWordDistance = 'dog cat hello cat dog dog hello cat world';
const word1 = 'hello';
const word2 = 'world';
console.log(`${findSmallestDistance(textToFindWordDistance, word1, word2)}`);
console.log('\n');

/*
 * Q40.
 * Given a tree where each edge has a weight, compute the length of the longest
 * path in the tree.
 * For example, given the following tree:
 * "    a      "
 * "   /|\     "
 * "  b c d    "
 * "     / \   "
 * "    e   f  "
 * "   / \     "
 * "  g   h    "
 * and the weights: a-b: 3, a-c: 5, a-d: 8, d-e: 2, d-f: 4, e-g: 1, e-h: 1, the
 * longest path would be c -> a -> d -> f, with a length of 17.
 * The path does not have to pass through the root, and each node can have any
 * amount of children.
 */
class GraphNode {
    constructor(id) {
        this.id = id;
        this.edges = [];
    }
}
class Edge {
    constructor(destination, weight) {
        this.destination = destination;
        this.weight = weight;
    }
}

let longestPath = 0;

function calculateLongestPath(root) {
    if (!root) {
        return 0;
    }

    calculatePath(root, null);

    return longestPath;
}

function calculatePath(node, parent) {
    if (node.edges.length === 1 && node !== parent) {
        return 0;
    }

    let maxPath1 = 0;
    let maxPath2 = 0;

    for (const edge of node.edges) {
        if (edge.destination !== parent) {
            const subPath = calculatePath(edge.destination, node) + edge.weight;
            if (subPath > maxPath1) {
                maxPath2 = maxPath1;
                maxPath1 = subPath;
            } else if (subPath > maxPath2) {
                maxPath2 = subPath;
            }
        }
    }
    longestPath = Math.max(longestPath, maxPath1 + maxPath2);
    return maxPath1;
}

console.log('========= Q40 =========');
const a = new GraphNode('a');
const b = new GraphNode('b');
const c = new GraphNode('c');
const d = new GraphNode('d');
const e = new GraphNode('e');
const f = new GraphNode('f');
const g = new GraphNode('g');
const h = new GraphNode('h');

a.edges.push(new Edge(b, 3));
a.edges.push(new Edge(c, 5));
a.edges.push(new Edge(d, 8));

d.edges.push(new Edge(e, 2));
d.edges.push(new Edge(f, 4));

e.edges.push(new Edge(g, 1));
e.edges.push(new Edge(h, 1));

console.log(`Longest Path Length: ${calculateLongestPath(a)}`);
console.log('\n');