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
        this.#_frequencyToKeys.set(1, this.#_frequencyToKeys.get(1) ? this.#_frequencyToKeys.get(1).add(key) : new Set([key]));
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
