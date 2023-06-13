/*
 * Q1.
 * Given the root to a binary tree, implement serialize(root), which serializes
 * the tree into a string, and deserialize(s), which deserializes the string
 * back into the tree.
 * For example, given the following Node class
 * "class Node:                                            "
 * "    def __init__(self, val, left=None, right=None):    "
 * "        self.val = val                                 "
 * "        self.left = left                               "
 * "        self.right = right                             "
 * The following test should pass:
 * node = Node('root', Node('left', Node('left.left')), Node('right'))
 * assert deserialize(serialize(node)).left.left.val == 'left.left'
 */
class Node {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function serialize(root) {
    if (!root) {
        return 'null';
    }

    const leftSerialized = serialize(root.left);
    const rightSerialized = serialize(root.right);

    return `${root.val},${leftSerialized},${rightSerialized}`;
}

function deserialize(data) {
    const nodes = data.split(',');
    return deserializeHelper(nodes);
}

function deserializeHelper(nodes) {
    const val = nodes.shift();
    if (val === 'null') {
        return null;
    }

    const node = new Node(val);
    node.left = deserializeHelper(nodes);
    node.right = deserializeHelper(nodes);

    return node;
}

console.log('========= Q1 =========');
const node = new Node(
    'root',
    new Node('left', new Node('left.left')),
    new Node('right')
);
const serialized = serialize(node);
console.log(`Serialized Tree: ${serialized}`);
const deserialized = deserialize(serialized);
console.log(`Deserialized Tree: ${deserialized.left.left.val}`);
console.log('\n');

/*
 * Q2.
 * cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
 * and last element of that pair. For example, car(cons(3, 4)) returns 3, and
 * cdr(cons(3, 4)) returns 4.
 * Given this implementation of cons:
 * "def cons(a, b):        "
 * "    def pair(f):       "
 * "        return f(a, b) "
 * "    return pair        "
 * Implement car and cdr.
 */
class Pair {
    constructor(first, second) {
        this.first = first;
        this.second = second;
    }

    car() {
        return this.first;
    }

    cdr() {
        return this.second;
    }
}

function cons(a, b) {
    return new Pair(a, b);
}

function car(pair) {
    return pair.car();
}

function cdr(pair) {
    return pair.cdr();
}

console.log('========= Q2 =========');
const pair = cons(3, 4);
console.log(`car(cons(3, 4)) = ${car(pair)}`);
console.log(`cdr(cons(3, 4)) = ${cdr(pair)}`);
console.log('\n');

/*
 * Q3.
 * Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the
 * number of ways it can be decoded.
 * For example, the message '111' would give 3, since it could be decoded as
 * 'aaa', 'ka', and 'ak'.
 * You can assume that the messages are decodable. For example, '001' is not
 * allowed.
 */
function numDecodings(message) {
    if (!message || message.length === 0) {
        return 0;
    }

    const n = message.length;
    let dp = new Array(n + 1).fill(0);
    dp[0] = 1;

    // Handle single-digit cases
    dp[1] = message[0] === '0' ? 0 : 1;

    for (let i = 2; i <= n; i++) {
        const oneDigit = parseInt(message.substring(i - 1, i));
        const twoDigits = parseInt(message.substring(i - 2, i));

        if (oneDigit >= 1 && oneDigit <= 9) {
            dp[i] += dp[i - 1];
        }

        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }

    return dp[n];
}

console.log('========= Q3 =========');
const message = '111';
console.log(`Number of ways to decode ${message}: ${numDecodings(message)}`);
console.log('\n');

/*
 * Q4.
 * Implement a job scheduler which takes in a function f and an integer n, and
 * calls f after n milliseconds.
 */
function scheduleJob(task, delayMillis) {
    setTimeout(task, delayMillis);
}

function myTask() {
    console.log('Job executed after delay');
}

console.log('========= Q4 =========');
const delayMillis = 3000;
scheduleJob(myTask, delayMillis);
console.log('Job scheduled');
console.log('\n');

/*
 * Q5.
 * Implement an autocomplete system. That is, given a query string s and a set
 * of all possible query strings, return all strings in the set that have s as a
 * prefix.
 * For example, given the query string de and the set of strings [dog, deer,
 * deal], return [deer, deal].
 * Hint: Try preprocessing the dictionary into a more efficient data structure
 * to speed up queries.
 */
class TrieNode {
    constructor() {
        this.children = Array(26).fill(null);
        // .map(() => new TrieNode());
        this._isEndOfWord = false;
    }

    getChild(ch) {
        return this.children[ch.charCodeAt(0) - 'a'.charCodeAt(0)];
    }

    setChild(ch, node) {
        this.children[ch.charCodeAt(0) - 'a'.charCodeAt(0)] = node;
    }

    get isEndOfWord() {
        return this._isEndOfWord;
    }

    set isEndOfWord(endOfWord) {
        this._isEndOfWord = endOfWord;
    }
}

class AutocompleteSystem {
    #root;

    constructor() {
        this.#root = new TrieNode();
    }

    insert(word) {
        let current = this.#root;

        for (const ch of word) {
            let child = current.getChild(ch);

            if (!child) {
                child = new TrieNode();
                current.setChild(ch, child);
            }
            current = child;
        }
        current.isEndOfWord = true;
    }

    search(prefix) {
        let results = [];
        let current = this.#root;

        for (const ch of prefix) {
            let child = current.getChild(ch);
            if (!child) {
                return results;
            }
            current = child;
        }

        this.collectWords(current, prefix, results);
        return results;
    }

    collectWords(node, prefix, results) {
        if (node.isEndOfWord) {
            results.push(prefix);
        }

        for (let ch = 'a'.charCodeAt(0); ch <= 'z'.charCodeAt(0); ch++) {
            const currentChar = String.fromCharCode(ch);
            let child = node.getChild(currentChar);
            if (child) {
                this.collectWords(child, prefix + currentChar, results);
            }
        }
    }
}

console.log('========= Q5 =========');
const autocomplete = new AutocompleteSystem();
autocomplete.insert('dog');
autocomplete.insert('deer');
autocomplete.insert('deal');

const query = 'de';
const autocompleteResults = autocomplete.search(query);
console.log(`Autocomplete results for ${query}: ${autocompleteResults}`);
console.log('\n');

/*
 * Q6.
 * The area of a circle is defined as πr^2. Estimate π to 3 decimal places using
 * a Monte Carlo method.
 * Hint: The basic equation of a circle is x2 + y2 = r2.
 */
function estimatePieDecimalPlaces() {
    const totalPoints = 1000000;
    let pointsInsideCircle = 0;

    for (let i = 0; i < totalPoints; i++) {
        const x = Math.random();
        const y = Math.random();

        const distance = x * x + y * y;
        if (distance <= 1) {
            pointsInsideCircle++;
        }
    }

    return (4 * pointsInsideCircle) / totalPoints;
}

console.log('========= Q6 =========');
console.log(`Estimated value of pi: ${estimatePieDecimalPlaces()}`);
console.log('\n');

/*
 * Q7.
 * Given a stream of elements too large to store in memory, pick a random
 * element from the stream with uniform probability.
 */
class RandomElementPicker {
    #currentElement;
    #count;

    constructor() {
        this.#currentElement = null;
        this.#count = 0;
    }

    pickElement(element) {
        this.#count++;

        // The probability of updating the currentElement decreases as more elements are
        // encountered, ensuring a uniform selection.
        if (Math.floor(Math.random() * this.#count) === 0) {
            this.#currentElement = element;
        }
    }

    getRandomElement() {
        return this.#currentElement;
    }
}

console.log('========= Q7 =========');
const picker = new RandomElementPicker();

for (let i = 0; i < 1000; i++) {
    picker.pickElement(i);
}

const randomElement = picker.getRandomElement();
console.log(`Random element: ${randomElement}`);
console.log('\n');

/*
 * Q8.
 * A builder is looking to build a row of N houses that can be of K different
 * colors. He has a goal of minimizing cost while ensuring that no two
 * neighboring houses are of the same color.
 * Given an N by K matrix where the nth row and kth column represents the cost
 * to build the nth house with kth color, return the minimum cost which achieves
 * this goal.
 */
function minCost(costs) {
    if (!costs || costs.length === 0) {
        return 0;
    }

    const n = costs.length;
    const k = costs[0].length;

    let dp = new Array(n).fill(0).map(() => new Array(k).fill(0));

    // Initialize the first row of the dp matrix with the costs of the first house
    for (let i = 0; i < k; i++) {
        dp[0][i] = costs[0][i];
    }

    for (let i = 1; i < n; i++) {
        for (let j = 0; j < k; j++) {
            dp[i][j] = costs[i][j] + getMinCost(dp, i - 1, j);
        }
    }

    //Find the minimum cost among the last row
    let minCost = dp[n - 1][0];
    for (let i = 1; i < k; i++) {
        minCost = Math.min(minCost, dp[n - 1][i]);
    }
    console.log(dp);
    return minCost;
}

// Helper method to get the minimum cost of painting the previous house with a
// different color
function getMinCost(dp, row, colour) {
    let minCost = Number.MAX_VALUE;

    for (let i = 0; i < dp[row].length; i++) {
        if (i !== colour) {
            minCost = Math.min(minCost, dp[row][i]);
        }
    }

    return minCost;
}

console.log('========= Q8 =========');
const costs = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
];
console.log(`Minimum cost: ${minCost(costs)}`);
console.log('\n');

/*
 * Q9.
 * Given a dictionary of words and a string made up of those words (no spaces),
 * return the original sentence in a list. If there is more than one possible
 * reconstruction, return any of them. If there is no possible reconstruction,
 * then return null.
 * For example, given the set of words 'quick', 'brown', 'the', 'fox', and the
 * string "thequickbrownfox", you should return ['the', 'quick', 'brown',
 * 'fox'].
 * Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the
 * string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or
 * ['bedbath', 'and', 'beyond'].
 */
function reconstructSentence(wordDict, sentence) {
    let memo = new Map();
    return reconstruct(wordDict, sentence, memo);
}

function reconstruct(wordDict, sentence, memo) {
    if (memo.has(sentence)) {
        return memo.get(sentence);
    }

    let result = [];
    if (sentence.length === 0) {
        result.push('');
        return result;
    }

    for (const word of wordDict) {
        if (sentence.startsWith(word)) {
            const suffix = sentence.substring(word.length);
            const subSentences = reconstruct(wordDict, suffix, memo);
            for (const subSentence of subSentences) {
                const reconstructed =
                    word + (subSentence.length === 0 ? '' : ' ') + subSentence;
                result.push(reconstructed);
            }
        }
    }
    memo.set(sentence, result);
    return result;
}

console.log('========= Q9 =========');
const wordDict1 = ['quick', 'brown', 'the', 'fox'];
const sentence1 = 'thequickbrownfox';
console.log(
    `Reconstructed Sentence 1: ${reconstructSentence(wordDict1, sentence1)}`
);

const wordDict2 = ['bed', 'bath', 'bedbath', 'and', 'beyond'];
const sentence2 = 'bedbathandbeyond';
console.log(
    `Reconstructed Sentence 2: ${reconstructSentence(wordDict2, sentence2)}`
);
console.log('\n');

/*
 * Q10.
 * Implement locking in a binary tree. A binary tree node can be locked or
 * unlocked only if all of its descendants or ancestors are not locked.
 * Design a binary tree node class with the following methods:
 * is_locked, which returns whether the node is locked
 * lock, which attempts to lock the node. If it cannot be locked, then it should
 * return false. Otherwise, it should lock it and return true.
 * unlock, which unlocks the node. If it cannot be unlocked, then it should
 * return false. Otherwise, it should unlock it and return true.
 * You may augment the node to add parent pointers or any other property you
 * would like. You may assume the class is used in a single-threaded program, so
 * there is no need for actual locks or mutexes. Each method should run in O(h),
 * where h is the height of the tree.
 */
class LockingTreeNode {
    #_locked;
    #_lockedDescendantsCount;
    #_parent;

    constructor(value = 0, parent = null, left = null, right = null) {
        this.value = value;
        this.#_locked = false;
        this.#_lockedDescendantsCount = 0;
        this.#_parent = parent;
        this.left = left;
        this.right = right;
    }

    get isLocked() {
        return this.#_locked;
    }

    lock() {
        if (this.#_lockedDescendantsCount > 0 || this.hasLockedAncestor()) {
            return false;
        }

        this.#_locked = true;

        this.updateLockedDescendantsCount(1);

        return true;
    }

    unlock() {
        if (this.#_lockedDescendantsCount > 0 || this.hasLockedAncestor()) {
            return false;
        }

        this.#_locked = false;

        this.updateLockedDescendantsCount(-1);

        return true;
    }

    hasLockedAncestor() {
        let current = this.#_parent;
        while (current) {
            if (current.isLocked) {
                return true;
            }
            current = current.#_parent;
        }

        return false;
    }

    updateLockedDescendantsCount(count) {
        let current = this.#_parent;
        while (current) {
            current.#_lockedDescendantsCount += count;
            current = current.#_parent;
        }
    }
}

console.log('========= Q10 =========');
const root = new LockingTreeNode();
let node1 = new LockingTreeNode(1, root);
let node2 = new LockingTreeNode(2, root);
let node3 = new LockingTreeNode(3, node1);
let node4 = new LockingTreeNode(4, node2);

root.left = node1;
root.right = node2;
node1.left = node3;
node2.right = node4;

console.log('Locking node 1', node1.lock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // true

console.log('Locking node 3', node3.lock()); // false
console.log(`Is node 3 locked? ${node3.isLocked}`); // false

console.log('Unlocking node 1', node1.unlock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // false

console.log('Locking node 2', node2.lock()); // true
console.log(`Is node 2 locked? ${node2.isLocked}`); // true

console.log('Locking node 1', node1.lock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // true

console.log('Unlocking node 2', node2.unlock()); // true
console.log(`Is node 2 locked? ${node2.isLocked}`); // false
console.log('\n');

/*
* Q11.
* Given a singly linked list and an integer k, remove the kth last element from
* the list. k is guaranteed to be smaller than the length of the list.
* The list is very long, so making more than one pass is prohibitively
* expensive.
* Do this in constant space and in one pass.
*/
class ListNode {
    constructor(value, next = null) {
        this.value = value;
        this.next = next;
    }
}

function removeKthLast(head, k) {
    let fast = head;
    let slow = head;

    for (let i = 0; i < k; i++) {
        fast = fast.next;
    }

    // If fast is null, it means k is equal to the length of the list
    // So we need to remove the head of the list
    if (!fast) {
        return head.next;
    }

    while (fast.next) {
        fast = fast.next;
        slow = slow.next;
    }

    slow.next = slow.next.next;

    return head;
}

console.log('========= Q11 =========');
const head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
const k = 2;
let result = removeKthLast(head, k);

while (result) {
    console.log(result.value);
    result = result.next;
}
console.log('\n');

/*
* Q12.
* Write an algorithm to justify text. Given a sequence of words and an integer
* line length k, return a list of strings which represents each line, fully
* justified.
* More specifically, you should have as many words as possible in each line.
* There should be at least one space between each word. Pad extra spaces when
* necessary so that each line has exactly length k. Spaces should be
* distributed as equally as possible, with the extra spaces, if any,
* distributed starting from the left.
* If you can only fit one word on a line, then you should pad the right-hand
* side with spaces.
* Each word is guaranteed not to be longer than k.
* For example, given the list of words ["the", "quick", "brown", "fox",
* "jumps", "over", "the", "lazy", "dog"] and k = 16, you should return the
* following:
* ["the  quick brown", # 1 extra space on the left
* "fox  jumps  over", # 2 extra spaces distributed evenly
* "the   lazy   dog"] # 4 extra spaces distributed evenly
*/
function justifyText(words, k) {
    let justifiedLines = [];
    const n = words.length;
    let i = 0;

    while (i < n) {
        let line = '';
        let lineLength = words[i].length;
        let wordCount = 1;
        let j = i + 1;

        while (j < n && lineLength + 1 + words[j].length <= k) {
            lineLength += 1 + words[j].length;
            wordCount++;
            j++;
        }

        let extraSpaces = k - lineLength;
        let spacePerWord = wordCount > 1 ? Math.floor(extraSpaces / (wordCount - 1)) : extraSpaces;

        line += words[i];

        for (let x = i + 1; x < j; x++) {
            for (let s = 0; s < spacePerWord; s++) {
                line += ' ';
            }

            if (x - i <= extraSpaces % (wordCount - 1)) {
                line += ' ';
            }

            line += ' ' + words[x];
        }

        if (wordCount === 1) {
            while (line.length < k) {
                line += ' ';
            }
        }

        justifiedLines.push(line);

        i = j;
    }

    return justifiedLines;
}

console.log('========= Q12 =========');
const words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
const lineLength = 16;

const justifiedLines = justifyText(words, lineLength);

for (const line of justifiedLines) {
    console.log(line);
}
console.log('\n');

/*
* Q13.
* You are given an array of non-negative integers that represents a
* two-dimensional elevation map where each element is unit-width wall and the
* integer is the height. Suppose it will rain and all spots between two walls
* get filled up.
* Compute how many units of water remain trapped on the map in O(N) time and
* O(1) space.
* For example, given the input [2, 1, 2], we can hold 1 unit of water in the
* middle.
* Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2
* in the second, and 3 in the fourth index (we cannot hold 5 since it would run
* off to the left), so we can trap 8 units of water.
*/
function trapWater(heights) {
    let left = 0;
    let right = heights.length - 1;
    let maxLeft = 0;
    let maxRight = 0;
    let totalWater = 0;

    while (left < right) {
        if (heights[left] <= heights[right]) {
            if (heights[left] >= maxLeft) {
                maxLeft = heights[left];
            } else {
                totalWater += maxLeft - heights[left];
            }
            left++;
        } else {
            if (heights[right] >= maxRight) {
                maxRight = heights[right];
            } else {
                totalWater += maxRight - heights[right];
            }
            right--;
        }
    }
    return totalWater
}

console.log('========= Q13 =========');
const heights1 = [2, 1, 2];
console.log(trapWater(heights1)); // 1
const heights2 = [3, 0, 1, 3, 0, 5];
console.log(trapWater(heights2)); // 8
console.log('\n');

/*
* Q14.
* Given a string, find the palindrome that can be made by inserting the fewest
* number of characters as possible anywhere in the word. If there is more than
* one palindrome of minimum length that can be made, return the
* lexicographically earliest one (the first one alphabetically).
* For example, given the string "race", you should return "ecarace", since we
* can add three letters to it (which is the smallest amount to make a
* palindrome). There are seven other palindromes that can be made from "race"
* by adding three letters, but "ecarace" comes first alphabetically.
* As another example, given the string "google", you should return "elgoogle".
*/
function findPalindrome(s) {
    const n = s.length;
    let dp = new Array(n + 1).fill(0).map(() => new Array(n + 1).fill(0));

    // the minimum number of insertions required to make the substring from index i
    // to j a palindrome.
    for (let len = 2; len <= n; len++) {
        for (let i = 0; i <= n - len + 1; i++) {
            let j = i + len - 1;

            if (s[i - 1] === s[j - 1]) { // Insertions are not needed
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
            }
        }
    }

    let palindrome = '';
    let i = 1;
    let j = n;

    while (i < j) {
        if (s.charAt(i - 1) === s.charAt(j - 1)) {
            palindrome += s.charAt(i - 1);
            i++;
            j--;
        } else if (dp[i][j - 1] <= dp[i + 1][j]) { // If left is smaller, add a ch at j
            palindrome += s.charAt(j - 1);
            j--;
        } else { // If down is smaller, add a ch at i
            palindrome += s.charAt(i - 1);
            i++;
        }
    }

    const endIndexToCopy = palindrome.length - 1;
    if (i === j) {
        palindrome += s.charAt(i - 1);
    }

    for (i = endIndexToCopy; i >= 0; i--) {
        palindrome += palindrome.charAt(i);
    }

    return palindrome;
}

console.log('========= Q14 =========');
const input1 = 'race';
console.log(findPalindrome(input1)); // ecarace
const input2 = 'google';
console.log(findPalindrome(input2)); // elgoogle
console.log('\n');

/*
* Q15.
* Given the root to a binary search tree, find the second largest node in the
* tree.
*/
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

function findSecondLargest(root) {
    if (!root) {
        throw new Error('Invalid input: BST is empty');
    }

    const largest = findLargest(root);

    if (largest.left) {
        return findLargest(largest.left);
    } else {
        return findParentOfLargest(root, largest);
    }
}

function findLargest(node) {
    while (node.right) {
        node = node.right;
    }
    return node;
}

function findParentOfLargest(root, largest) {
    let parent = null;
    let current = root;

    while (current && current !== largest) {
        parent = current;
        current = current.right;
    }
    return parent;
}

console.log('========= Q15 =========');
const binaryTreeRoot = new TreeNode(6);
binaryTreeRoot.left = new TreeNode(2);
binaryTreeRoot.right = new TreeNode(8);
binaryTreeRoot.left.left = new TreeNode(1);
binaryTreeRoot.left.right = new TreeNode(4);
binaryTreeRoot.right.left = new TreeNode(7);
binaryTreeRoot.right.right = new TreeNode(9);

const secondLargest = findSecondLargest(binaryTreeRoot);
console.log(`Second largest node: ${secondLargest.val}`); // 8
console.log('\n');

/*
* Q16.
* Conway's Game of Life takes place on an infinite two-dimensional board of
* square cells. Each cell is either dead or alive, and at each tick, the
* following rules apply:
* Any live cell with less than two live neighbours dies.
* Any live cell with two or three live neighbours remains living.
* Any live cell with more than three live neighbours dies.
* Any dead cell with exactly three live neighbours becomes a live cell.
* A cell neighbours another cell if it is horizontally, vertically, or
* diagonally adjacent.
* Implement Conway's Game of Life. It should be able to be initialized with a
* starting list of live cell coordinates and the number of steps it should run
* for. Once initialized, it should print out the board state at each step.
* Since it's an infinite board, print out only the relevant coordinates, i.e.
* from the top-leftmost live cell to bottom-rightmost live cell.
* You can represent a live cell with an asterisk (*) and a dead cell with a dot
* (.).
*/
class Cell {
    #_x;
    #_y;
    #_alive;

    constructor(x, y, alive) {
        this.#_x = x;
        this.#_y = y;
        this.#_alive = alive;
    }

    get x() {
        return this.#_x;
    }

    get y() {
        return this.#_y;
    }

    get alive() {
        return this.#_alive;
    }

    set alive(alive) {
        this.#_alive = alive;
    }
}
class GameOfLife {
    #_board;

    constructor(size) {
        this.#_board = new Array(size)
            .fill(null)
            .map(() => new Array(size).fill(null));

        for (let i = 0; i < this.#_board.length; i++) {
            for (let j = 0; j < this.#_board.length; j++) {
                this.#_board[i][j] = new Cell(i, j, false);
            }
        }
    }

    initialize(liveCellCoordinates) {
        for (const coordinate of liveCellCoordinates) {
            const x = coordinate[0];
            const y = coordinate[1];
            this.#_board[x][y] = new Cell(x, y, true);
        }
    }

    run(steps) {
        for (let step = 1; step <= steps; step++) {
            console.log(`Step ${step}:`);
            this.printBoard();

            let nextBoard = new Array(this.#_board.length)
                .fill(null)
                .map(() => new Array(this.#_board.length).fill(null));

            for (let i = 0; i < this.#_board.length; i++) {
                for (let j = 0; j < this.#_board.length; j++) {
                    nextBoard[i][j] = this.getNextCellState(i, j);
                }
            }

            this.#_board = nextBoard;
        }
    }

    getNextCellState(x, y) {
        let liveNeighbours = this.countLiveNeighbours(x, y);
        let currentCell = this.#_board[x][y];
        let nextCell = new Cell(x, y, currentCell.alive);

        if (currentCell.alive) {
            if (liveNeighbours < 2 || liveNeighbours > 3) {
                nextCell.alive = false;
            }
        } else {
            if (liveNeighbours === 3) {
                nextCell.alive = true;
            }
        }
        return nextCell;
    }

    countLiveNeighbours(x, y) {
        let count = 0;

        for (let i = x - 1; i <= x + 1; i++) {
            for (let j = y - 1; j <= y + 1; j++) {
                if (i === x && j === y) {
                    continue;
                }

                if (
                    this.isValidCoordinate(i, j) &&
                    this.#_board[i][j] &&
                    this.#_board[i][j].alive
                ) {
                    count++;
                }
            }
        }
        return count;
    }

    isValidCoordinate(x, y) {
        return (
            x >= 0 &&
            x < this.#_board.length &&
            y >= 0 &&
            y < this.#_board.length
        );
    }

    printBoard() {
        let minX = Number.MAX_SAFE_INTEGER;
        let maxX = Number.MIN_SAFE_INTEGER;
        let minY = Number.MAX_SAFE_INTEGER;
        let maxY = Number.MIN_SAFE_INTEGER;

        for (const row of this.#_board) {
            for (const cell of row) {
                if (cell && cell.alive) {
                    minX = Math.min(minX, cell.x);
                    maxX = Math.max(maxX, cell.x);
                    minY = Math.min(minY, cell.y);
                    maxY = Math.max(maxY, cell.y);
                }
            }
        }

        let stringToPrint = '';
        for (let i = minX; i <= maxX; i++) {
            for (let j = minY; j <= maxY; j++) {
                const cell = this.#_board[i][j];
                const symbol = cell && cell.alive ? '*' : '.';
                stringToPrint += symbol;
            }
            stringToPrint += '\n';
        }
        console.log(stringToPrint);
    }
}

console.log('========= Q16 =========');
const liveCellCoordinates = [
    [1, 2],
    [2, 2],
    [2, 3],
    [3, 1],
    [3, 2],
];

const size = 5;
const steps = 5;

const game = new GameOfLife(size);
game.initialize(liveCellCoordinates);
game.run(steps);

/*
 * Q17.
 * Given an unordered list of flights taken by someone, each represented as
 * (origin, destination) pairs, and a starting airport, compute the person's
 * itinerary. If no such itinerary exists, return null. If there are multiple
 * possible itineraries, return the lexicographically smallest one. All flights
 * must be used in the itinerary.
 * For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'),
 * ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting airport 'YUL', you should return
 * the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].
 * Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting
 * airport 'COM', you should return null.
 * Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
 * and starting airport 'A', you should return the list ['A', 'B', 'C', 'A',
 * 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary.
 * However, the first one is lexicographically smaller.
 */
function findItinerary(flights, startAirport) {
    let flightMap = new Map();

    for (const flight of flights) {
        const origin = flight[0];
        const destination = flight[1];
        if (!flightMap.has(origin)) {
            flightMap.set(origin, []);
        }
        flightMap.get(origin).push(destination);
    }

    let itinerary = [];
    dfs(flightMap, startAirport, itinerary);

    if (itinerary.length !== flights.length + 1) {
        return null;
    }

    return itinerary;
}

function dfs(flightMap, airport, itinerary) {
    let destinations = flightMap.get(airport);
    if (destinations) {
        destinations.sort();
    }

    while (destinations && destinations.length > 0) {
        const nextAirport = destinations.shift();
        dfs(flightMap, nextAirport, itinerary);
    }
    itinerary.unshift(airport);
}

console.log('========= Q17 =========');
const flights1 = [
    ['SFO', 'HKO'],
    ['YYZ', 'SFO'],
    ['YUL', 'YYZ'],
    ['HKO', 'ORD'],
];
const startAirport1 = 'YUL';
console.log(`Itinerary 1: ${findItinerary(flights1, startAirport1)}`);

const flights2 = [
    ['SFO', 'COM'],
    ['COM', 'YYZ'],
];
const startAirport2 = 'COM';
console.log(`Itinerary 2: ${findItinerary(flights2, startAirport2)}`);

const flights3 = [
    ['A', 'B'],
    ['A', 'C'],
    ['B', 'C'],
    ['C', 'A'],
];
const startAirport3 = 'A';
console.log(`Itinerary 3: ${findItinerary(flights3, startAirport3)}`);
console.log('\n');

/*
 * Q18.
 * We can determine how "out of order" an array A is by counting the number of
 * inversions it has. Two elements A[i] and A[j] form an inversion if A[i] >
 * A[j] but i < j. That is, a smaller element appears after a larger element.
 * Given an array, count the number of inversions it has. Do this faster than
 * O(N^2) time.
 * You may assume each element in the array is distinct.
 * For example, a sorted list has zero inversions. The array [2, 4, 1, 3, 5] has
 * three inversions: (2, 1), (4, 1), and (4, 3). The array [5, 4, 3, 2, 1] has
 * ten inversions: every distinct pair forms an inversion.
 */
function countInversion(arr) {
    if (!arr || arr.length <= 1) {
        return 0;
    }

    let temp = new Array(arr.length);
    return mergeSortAndCount(arr, temp, 0, arr.length - 1);
}

function mergeSortAndCount(arr, temp, left, right) {
    let count = 0;
    if (left < right) {
        const mid = left + Math.floor((right - left) / 2);
        count += mergeSortAndCount(arr, temp, left, mid);
        count += mergeSortAndCount(arr, temp, mid + 1, right);
        count += merge(arr, temp, left, mid + 1, right);
    }
    return count;
}

function merge(arr, temp, left, mid, right) {
    let i = left;
    let j = mid;
    let k = left;
    let count = 0;

    // Compare two arrays and merge them in sorted order
    while (i <= mid - 1 && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            count += mid - i; // Count the number of inversions
        }
    }

    // Copy the remaining elements of left array
    while (i <= mid - 1) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
    return count;
}

console.log('========= Q18 =========');
const arr1 = [2, 4, 1, 3, 5];
console.log(`Inversion count of ${arr1}: ${countInversion(arr1)}`);

const arr2 = [5, 4, 3, 2, 1];
console.log(`Inversion count of ${arr2}: ${countInversion(arr2)}`);
console.log('\n');

/*
 * Q19.
 * Given pre-order and in-order traversals of a binary tree, write a function to
 * reconstruct the tree.
 * For example, given the following preorder traversal:
 * [a, b, d, e, c, f, g]
 * And the following inorder traversal:
 * [d, b, e, a, f, c, g]
 * You should return the following tree:
 * "    a      "
 * "   / \     "
 * "  b   c    "
 * " / \ / \   "
 * "d  e f  g  "
 */
function buildTree(preorder, inorder) {
    if (!preorder || !inorder || preorder.length !== inorder.length) {
        return null;
    }

    return buildTreeHelper(
        preorder,
        0,
        preorder.length - 1,
        inorder,
        0,
        inorder.length - 1
    );
}

function buildTreeHelper(preorder, preStart, preEnd, inorder, inStart, inEnd) {
    if (preStart > preEnd || inStart > inEnd) {
        return null;
    }

    const rootVal = preorder[preStart];
    const root = new TreeNode(rootVal);

    let rootIndexInorder = -1;
    for (let i = inStart; i <= inEnd; i++) {
        if (inorder[i] === root.val) {
            rootIndexInorder = i;
            break;
        }
    }

    let leftSubtreeSize = rootIndexInorder - inStart;
    root.left = buildTreeHelper(
        preorder,
        preStart + 1,
        preStart + leftSubtreeSize,
        inorder,
        inStart,
        rootIndexInorder - 1
    );
    root.right = buildTreeHelper(
        preorder,
        preStart + leftSubtreeSize + 1,
        preEnd,
        inorder,
        rootIndexInorder + 1,
        inEnd
    );

    return root;
}

function printInorder(root) {
    if (!root) {
        return;
    }

    printInorder(root.left);
    console.log(root.val);
    printInorder(root.right);
}

console.log('========= Q19 =========');
const preorder = ['a', 'b', 'd', 'e', 'c', 'f', 'g'];
const inorder = ['d', 'b', 'e', 'a', 'f', 'c', 'g'];

const constructedTreeRoot = buildTree(preorder, inorder);
printInorder(constructedTreeRoot);
console.log('\n');

/*
 * Q20.
 * Given an array of numbers, find the maximum sum of any contiguous subarray of
 * the array.
 * For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would
 * be 137, since we would take elements 42, 14, -5, and 86.
 * Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would
 * not take any elements.
 * Do this in O(N) time.
 */
function findMaxSubarraySum(nums) {
    let maxSum = 0;
    let currentSum = 0;

    for (const num of nums) {
        currentSum = Math.max(num, currentSum + num);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}

console.log('========= Q20 =========');
const arrToFindMaxSubarray1 = [34, -50, 42, 14, -5, 86];
const arrToFindMaxSubarray2 = [-5, -1, -8, -9];

console.log(
    `Maximum sum in arr1: ${findMaxSubarraySum(arrToFindMaxSubarray1)}`
);
console.log(
    `Maximum sum in arr2: ${findMaxSubarraySum(arrToFindMaxSubarray2)}`
);
console.log('\n');

/*
 * Q21.
 * Given a function that generates perfectly random numbers between 1 and k
 * (inclusive), where k is an input, write a function that shuffles a deck of
 * cards represented as an array using only swaps.
 * It should run in O(N) time.
 * Hint: Make sure each one of the 52! permutations of the deck is equally
 * likely.
 */
function shuffleDeck(deck) {
    const n = deck.length;

    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        swap(deck, i, j);
    }
}

function swap(deck, i, j) {
    let temp = deck[i];
    deck[i] = deck[j];
    deck[j] = temp;
}

console.log('========= Q21 =========');
let deck = new Array(52);
for (let i = 0; i < 52; i++) {
    deck[i] = i + 1;
}

shuffleDeck(deck);

console.log(`Shuffled deck: ${deck}`);
console.log('\n');

/*
 * Q22.
 * Implement a queue using two stacks. Recall that a queue is a FIFO (first-in,
 * first-out) data structure with the following methods: enqueue, which inserts
 * an element into the queue, and dequeue, which removes it.
 */
class Stack {
    #_stack;

    constructor() {
        this.#_stack = [];
    }

    push(element) {
        this.#_stack.push(element);
    }

    pop() {
        return this.#_stack.pop();
    }

    isEmpty() {
        return this.#_stack.length === 0;
    }

    get length() {
        return this.#_stack.length;
    }

    get elements() {
        return this.#_stack;
    }
}

class QueueUsingStack {
    #_enqueueStack;
    #_dequeueStack;

    constructor() {
        this.#_enqueueStack = new Stack();
        this.#_dequeueStack = new Stack();
    }

    enqueue(element) {
        this.#_enqueueStack.push(element);
    }

    dequeue() {
        if (this.#_dequeueStack.isEmpty()) {
            while (!this.#_enqueueStack.isEmpty()) {
                this.#_dequeueStack.push(this.#_enqueueStack.pop());
            }
        }

        return this.#_dequeueStack.pop();
    }

    isEmpty() {
        return this.#_enqueueStack.isEmpty() && this.#_dequeueStack.isEmpty();
    }

    size() {
        return this.#_enqueueStack.length + this.#_dequeueStack.length;
    }
}

console.log('========= Q22 =========');
const queue = new QueueUsingStack();
queue.enqueue(1);
queue.enqueue(2);
queue.enqueue(3);

console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 1
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 2

queue.enqueue(4);
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 3
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 4

console.log(`Is queue empty? ${queue.isEmpty()}`); // Output: true
console.log('\n');

/*
 * Q23.
 * Given an undirected graph represented as an adjacency matrix and an integer
 * k, write a function to determine whether each vertex in the graph can be
 * colored such that no two adjacent vertices share the same color using at most
 * k colors.
 */
class GraphColoring {
    #_graph;
    #_colors;
    #_numVertices;

    constructor(graph) {
        this.#_graph = graph;
        this.#_numVertices = graph.length;
        this.#_colors = new Array(this.#_numVertices).fill(0);
    }

    canColorGraph(k) {
        return this.canColorVertex(0, k);
    }

    canColorVertex(vertex, k) {
        if (vertex === this.#_numVertices) {
            return true; // All vertices have been colored
        }

        for (let color = 1; color <= k; color++) {
            if (this.isColorValid(vertex, color)) {
                this.#_colors[vertex] = color;

                if (this.canColorVertex(vertex + 1, k)) {
                    return true;
                }

                this.#_colors[vertex] = 0; // Backtrack and try a different color
            }
        }

        return false;
    }

    isColorValid(vertex, color) {
        for (let i = 0; i < this.#_numVertices; i++) {
            if (this.#_graph[vertex][i] === 1 && color === this.#_colors[i]) {
                return false;
            }
        }

        return true;
    }
}

console.log('========= Q23 =========');
const graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
];

const numOfColours = 3;
const graphColouring = new GraphColoring(graph);
console.log(
    `Can color graph with ${numOfColours} colours? ${graphColouring.canColorGraph(
        numOfColours
    )}`
);
console.log('\n');

/*
 * Q24.
 * Given a string s and an integer k, break up the string into multiple lines
 * such that each line has a length of k or less. You must break it up so that
 * words don't break across lines. Each line has to have the maximum possible
 * amount of words. If there's no way to break the text up, then return null.
 * You can assume that there are no spaces at the ends of the string and that
 * there is exactly one space between each word.
 * For example, given the string "the quick brown fox jumps over the lazy dog"
 * and k = 10, you should return: ["the quick", "brown fox", "jumps over",
 * "the lazy", "dog"]. No string in the list has a length of more than 10.
 */
function breakLines(s, k) {
    const words = s.split(' ');
    const lines = [];
    let currentLine = '';
    let currentLineLength = 0;

    for (const word of words) {
        const wordLength = word.length;

        if (currentLineLength + wordLength <= k) {
            currentLine += `${word} `;
            currentLineLength += wordLength + 1;
        } else {
            lines.push(currentLine.trim());
            currentLine = `${word} `;
            currentLineLength = wordLength + 1;
        }
    }

    lines.push(currentLine.trim());

    return lines;
}

console.log('========= Q24 =========');
const s = 'the quick brown fox jumps over the lazy dog';
const lineWidth = 10;

const lines = breakLines(s, lineWidth);
for (const line of lines) {
    console.log(line);
}
console.log('\n');

/*
 * Q25.
 * An sorted array of integers was rotated an unknown number of times.
 * Given such an array, find the index of the element in the array in faster
 * than linear time. If the element doesn't exist in the array, return null.
 * For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return
 * 4 (the index of 8 in the array).
 * You can assume all the integers in the array are unique.
 */
function search(nums, target) {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        let mid = left + Math.floor((right - left) / 2);

        if (nums[mid] === target) {
            return mid;
        }

        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return null;
}

console.log('========= Q25 =========');
const nums = [13, 18, 25, 2, 8, 10];
const target = 8;
console.log(`Index of ${target} is ${search(nums, target)}`);
console.log('\n');

/*
 * Q26.
 * Given a multiset of integers, return whether it can be partitioned into two
 * subsets whose sums are the same.
 * For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return
 * true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which
 * both add up to 55.
 * Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't
 * split it up into two subsets that add up to the same sum.
 */
function canPartition(nums) {
    let totalSum = 0;
    for (const num of nums) {
        totalSum += num;
    }

    if (totalSum % 2 !== 0) {
        return false; // If the total sum is odd, it cannot be partitioned into two equal subsets
    }

    let targetSum = totalSum / 2;
    const n = nums.length;
    let dp = new Array(n + 1).fill(false).map(() => new Array(targetSum + 1));

    // Initialize the first column with true, as we can make a sum of 0 with an
    // empty subset
    for (let i = 0; i <= n; i++) {
        dp[i][0] = true;
    }

    // Fill the dp array using the subset sum bottom-up approach
    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= targetSum; j++) {
            if (j < nums[i - 1]) {
                dp[i][j] = dp[i - 1][j]; // Copy previous number's result
            } else {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]]; // Copy previous number's result or use the result without previous number
            }
        }
    }

    return dp[n][targetSum];
}

console.log('========= Q26 =========');
const numsToPartition = [15, 5, 20, 10, 35, 15, 10];
console.log(
    `Can partition ${numsToPartition}? ${canPartition(numsToPartition)}`
);
console.log('\n');

/*
 * Q27.
 * Implement integer exponentiation. That is, implement the pow(x, y) function,
 * where x and y are integers and returns x^y.
 * Do this faster than the naive method of repeated multiplication.
 * For example, pow(2, 10) should return 1024.
 */
function pow(x, y) {
    if (y < 0) {
        return pow(1 / x, -y);
    } else if (y === 0) {
        return 1;
    } else if (y === 1) {
        return x;
    } else if (y % 2 === 0) {
        const halfPower = pow(x, y / 2);
        return halfPower * halfPower;
    } else {
        const halfPower = pow(x, (y - 1) / 2);
        return halfPower * halfPower * x;
    }
}

console.log('========= Q27 =========');
const x = 2;
const y = 10;
console.log(`${x}^${y} = ${pow(x, y)}`);
console.log('\n');

/*
 * Q28.
 * There is an N by M matrix of zeroes. Given N and M, write a function to count
 * the number of ways of starting at the top-left corner and getting to the
 * bottom-right corner. You can only move right or down.
 * For example, given a 2 by 2 matrix, you should return 2, since there are two
 * ways to get to the bottom-right:
 * Right, then down
 * Down, then right
 * Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
 */
function countWays(N, M) {
    let dp = new Array(N).fill(0).map(() => new Array(M).fill(0));

    for (let i = 0; i < N; i++) {
        dp[i][0] = 1;
    }
    for (let j = 0; j < M; j++) {
        dp[0][j] = 1;
    }

    for (let i = 1; i < N; i++) {
        for (let j = 1; j < M; j++) {
            // The number of ways to reach cell (i, j) is the sum of the ways
            // to reach the cell above (i-1, j) and the cell to the left (i, j-1)
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    return dp[N - 1][M - 1];
}

console.log('========= Q28 =========');
console.log(
    `Number of ways to reach bottom-right from top-left in 2x2 matrix: ${countWays(
        2,
        2
    )}`
);
console.log(
    `Number of ways to reach bottom-right from top-left in 5x5 matrix: ${countWays(
        5,
        5
    )}`
);
console.log('\n');

/*
 * Q29.
 * Assume you have access to a function toss_biased() which returns 0 or 1 with
 * a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know
 * the bias of the coin.
 * Write a function to simulate an unbiased coin toss.
 */
function tossUnbiased() {
    while (true) {
        const toss1 = tossBiased();
        const toss2 = tossBiased();
        if (toss1 !== toss2) {
            return toss1;
        }
    }
}

function tossBiased() {
    const probabilityOffHeads = 0.3;

    return Math.random() < probabilityOffHeads ? 1 : 0;
}

console.log('========= Q29 =========');
console.log(`Tossing unbiased coin: `);
for (let i = 0; i < 5; i++) {
    console.log(tossUnbiased());
}
console.log('\n');

/*
 * Q30.
 * On our special chessboard, two bishops attack each other if they share the
 * same diagonal. This includes bishops that have another bishop located between
 * them, i.e. bishops can attack through pieces.
 * You are given N bishops, represented as (row, column) tuples on a M by M
 * chessboard. Write a function to count the number of pairs of bishops that
 * attack each other. The ordering of the pair doesn't matter: (1, 2) is
 * considered the same as (2, 1).
 * For example, given M = 5 and the list of bishops:
 * (0, 0)
 * (1, 2)
 * (2, 2)
 * (4, 0)
 * The board would look like this:
 * [b 0 0 0 0]
 * [0 0 b 0 0]
 * [0 0 b 0 0]
 * [0 0 0 0 0]
 * [b 0 0 0 0]
 * You should return 2, since bishops 1 and 3 attack each other, as well as
 * bishops 3 and 4.
 */
function countAttackingPairs(bishops, M) {
    let positiveSlopes = new Map();
    let negativeSlopes = new Map();
    let pairs = 0;

    for (const bishop of bishops) {
        const row = bishop[0];
        const col = bishop[1];

        // Calculate the positive and negative diagonal slopes
        const positiveSlope = row + col;
        const negativeSlope = row - col;

        positiveSlopes.set(
            positiveSlope,
            (positiveSlopes.get(positiveSlope) || 0) + 1
        );
        negativeSlopes.set(
            negativeSlope,
            (negativeSlopes.get(negativeSlope) || 0) + 1
        );
    }

    for (const count of positiveSlopes.values()) {
        pairs += countPairs(count);
    }
    for (const count of negativeSlopes.values()) {
        pairs += countPairs(count);
    }

    return pairs;
}

function countPairs(count) {
    // Calculate the number of pairs using combination formula (nC2)
    return (count * (count - 1)) / 2;
}

console.log('========= Q30 =========');
const M = 5;
const bishops = [
    [0, 0],
    [1, 2],
    [2, 2],
    [4, 0],
];
console.log(`Number of attacking pairs: ${countAttackingPairs(bishops, M)}`);
console.log('\n');

/*
 * Q31.
 * Suppose you have a multiplication table that is N by N. That is, a 2D array
 * where the value at the i-th row and j-th column is (i + 1) * (j + 1) (if
 * 0-indexed) or i * j (if 1-indexed).
 * Given integers N and X, write a function that returns the number of times X
 * appears as a value in an N by N multiplication table.
 * For example, given N = 6 and X = 12, you should return 4, since the
 * multiplication table looks like this:
 * | 1 | 2 | 3 | 4 | 5 | 6 |
 * | 2 | 4 | 6 | 8 | 10 | 12 |
 * | 3 | 6 | 9 | 12 | 15 | 18 |
 * | 4 | 8 | 12 | 16 | 20 | 24 |
 * | 5 | 10 | 15 | 20 | 25 | 30 |
 * | 6 | 12 | 18 | 24 | 30 | 36 |
 * And there are 4 12's in the table.
 */
function countOccurrences(N, X) {
    let count = 0;

    for (let i = 1; i <= N; i++) {
        if (X % i === 0 && X / i <= N) {
            count++;
        }
    }

    return count;
}

console.log('========= Q31 =========');
const N = 6;
const X = 12;
console.log(
    `Number of occurrences of ${X} in ${N}x${N} multiplication table: ${countOccurrences(
        N,
        X
    )}`
);
console.log('\n');

/*
 * Q32.
 * You are given an N by M 2D matrix of lowercase letters. Determine the minimum
 * number of columns that can be removed to ensure that each row is ordered from
 * top to bottom lexicographically. That is, the letter at each column is
 * lexicographically later as you go down each row. It does not matter whether
 * each row itself is ordered lexicographically.
 * For example, given the following table:
 * cba
 * daf
 * ghi
 * This is not ordered because of the a in the center. We can remove the second
 * column to make it ordered:
 * ca
 * df
 * gi
 * So your function should return 1, since we only needed to remove 1 column.
 * As another example, given the following table:
 * abcdef
 * Your function should return 0, since the rows are already ordered (there's
 * only one row).
 * As another example, given the following table:
 * zyx
 * wvu
 * tsr
 * Your function should return 3, since we would need to remove all the columns
 * to order it.
 */
function minColumnRemovals(matrix) {
    if (!matrix || !matrix.length || matrix[0].length === 0) {
        return 0;
    }

    let rowCount = matrix.length;
    let colCount = matrix[0].length;
    let removalCount = 0;

    for (let col = 0; col < colCount; col++) {
        for (let row = 1; row < rowCount; row++) {
            if (matrix[row][col] < matrix[row - 1][col]) {
                removalCount++;
                break;
            }
        }
    }
    return removalCount;
}

console.log('========= Q32 =========');
const matrix1 = [
    ['c', 'b', 'a'],
    ['d', 'a', 'f'],
    ['g', 'h', 'i'],
];
console.log(`Minimum column removals: ${minColumnRemovals(matrix1)}`);

const matrix2 = [['a', 'b', 'c', 'd', 'e', 'f']];
console.log(`Minimum column removals: ${minColumnRemovals(matrix2)}`);

const matrix3 = [
    ['z', 'y', 'x'],
    ['w', 'v', 'u'],
    ['t', 's', 'r'],
];
console.log(`Minimum column removals: ${minColumnRemovals(matrix3)}`);
console.log('\n');

/*
 * Q33.
 * Given k sorted singly linked lists, write a function to merge all the lists
 * into one sorted singly linked list.
 */
function mergeKLists(lists) {
    if (!lists || !lists.length) {
        return null;
    }

    let queue = [];
    for (const head of lists) {
        if (head) {
            queue.push(head);
            queue.sort((a, b) => a.value - b.value);
        }
    }

    let dummy = new ListNode(0);
    let curr = dummy;

    while (queue.length) {
        let node = queue.shift();
        curr.next = node;
        curr = curr.next;

        if (node.next) {
            queue.push(node.next);
            queue.sort((a, b) => a.value - b.value);
        }
    }

    return dummy.next;
}

function printList(head) {
    let curr = head;
    let str = '';

    while (curr) {
        console.log(curr);
        str += curr.value + ' -> ';
        curr = curr.next;
    }
    console.log(str.substring(0, str.length - 3));
}

console.log('========= Q33 =========');
const list1 = new ListNode(1);
list1.next = new ListNode(4);
list1.next.next = new ListNode(5);

const list2 = new ListNode(1);
list2.next = new ListNode(3);
list2.next.next = new ListNode(4);

const list3 = new ListNode(2);
list3.next = new ListNode(6);

const lists = [list1, list2, list3];
const mergedList = mergeKLists(lists);
printList(mergedList);
console.log('\n');

/*
 * Q34.
 * Given an array of integers, write a function to determine whether the array
 * could become non-decreasing by modifying at most 1 element.
 * For example, given the array [10, 5, 7], you should return true, since we can
 * modify the 10 into a 1 to make the array non-decreasing.
 * Given the array [10, 5, 1], you should return false, since we can't modify
 * any one element to get a non-decreasing array.
 */
function checkPossibility(nums) {
    let count = 0;

    for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i] > nums[i + 1]) {
            count++;
            if (count > 1) {
                return false;
            }
            // Check if we can modify the current element or the next element
            if (i > 0 && nums[i - 1] > nums[i + 1]) {
                nums[i + 1] = nums[i];
            } else {
                nums[i] = nums[i + 1];
            }
        }
    }
    return true;
}

console.log('========= Q34 =========');
const nums1 = [10, 5, 7];
console.log(`Can be non-decreasing: ${checkPossibility(nums1)}`);

const nums2 = [10, 5, 1];
console.log(`Can be non-decreasing: ${checkPossibility(nums2)}`);
console.log('\n');

/*
 * Q35.
 * Invert a binary tree.
 * For example, given the following tree:
 * "    a      "
 * "   / \     "
 * "  b   c    "
 * " / \  /    "
 * "d   e f    "
 * should become:
 * "  a        "
 * " / \       "
 * " c  b      "
 * " \  / \    "
 * "  f e  d   "
 */
function invertTree(root) {
    if (!root) {
        return null;
    }

    // Swap the left and right children of the current node
    let temp = root.left;
    root.left = root.right;
    root.right = temp;

    invertTree(root.left);
    invertTree(root.right);

    return root;
}

function printTree(root) {
    if (!root) {
        return;
    }

    console.log(root.val);
    printTree(root.left);
    printTree(root.right);
}

console.log('========= Q35 =========');
let a = new TreeNode('a');
let b = new TreeNode('b');
let c = new TreeNode('c');
let d = new TreeNode('d');
let e = new TreeNode('e');
let f = new TreeNode('f');

a.left = b;
a.right = c;
b.left = d;
b.right = e;
c.left = f;

const inverted = invertTree(a);
printTree(inverted);
console.log('\n');

/*
 * Q36.
 * Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A
 * 1 represents land and 0 represents water, so an island is a group of 1s that
 * are neighboring whose perimeter is surrounded by water.
 * For example, this matrix has 4 islands.
 * 1 0 0 0 0
 * 0 0 1 1 0
 * 0 1 1 0 0
 * 0 0 0 0 0
 * 1 1 0 0 1
 * 1 1 0 0 1
 */
function countIslands(matrix) {
    let count = 0;
    const rows = matrix.length;
    const cols = matrix[0].length;
    let visited = new Array(rows)
        .fill(false)
        .map(() => new Array(cols).fill(false));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (matrix[i][j] === 1 && !visited[i][j]) {
                exploreIsland(matrix, visited, i, j);
                count++;
            }
        }
    }
    return count;
}

function exploreIsland(matrix, visited, row, col) {
    const rows = matrix.length;
    const cols = matrix[0].length;

    if (
        row < 0 ||
        row >= rows ||
        col < 0 ||
        col >= cols ||
        matrix[row][col] === 0 ||
        visited[row][col]
    ) {
        return;
    }

    visited[row][col] = true;

    exploreIsland(matrix, visited, row - 1, col);
    exploreIsland(matrix, visited, row + 1, col);
    exploreIsland(matrix, visited, row, col - 1);
    exploreIsland(matrix, visited, row, col + 1);
}

console.log('========= Q36 =========');
const matrix = [
    [1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
];
console.log(`Number of islands: ${countIslands(matrix)}`);
console.log('\n');

/*
 * Q37.
 * Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0,
 * using only mathematical or bit operations. You can assume b can only be 1 or
 * 0.
 */
function select(x, y, b) {
    // If b = 1, all bits will be 1 and return x
    // If b = 0, change all bits to 1 to return y
    return (x & -b) | (y & -(b ^ 1));
}

console.log('========= Q37 =========');
const xForOperation = 10;
const yForOperation = 20;
const bForOperation = 1;
console.log(`${select(xForOperation, yForOperation, bForOperation)}`); // Output: 10
console.log('\n');

/*
 * Q38.
 * Given a string of parentheses, write a function to compute the minimum number
 * of parentheses to be removed to make the string valid (i.e. each open
 * parenthesis is eventually closed).
 * For example, given the string "()())()", you should return 1. Given the
 * string ")(", you should return 2, since we must remove all of them.
 */
function minRemoval(s) {
    let count = 0;
    let stack = [];

    for (const c of s) {
        if (c === '(') {
            stack.push(c);
        } else if (c === ')') {
            if (stack.length > 0 && stack[stack.length - 1] === '(') {
                stack.pop();
            } else {
                count++;
            }
        }
    }
    count += stack.length;
    return count;
}

console.log('========= Q38 =========');
const parenthesesStr1 = '()())()';
console.log(
    `Minimum number of parentheses to be removed: ${minRemoval(
        parenthesesStr1
    )}`
);

const parenthesesStr2 = ')(';
console.log(
    `Minimum number of parentheses to be removed: ${minRemoval(
        parenthesesStr2
    )}`
);
console.log('\n');

/*
 * Q39.
 * Implement division of two positive integers without using the division,
 * multiplication, or modulus operators. Return the quotient as an integer,
 * ignoring the remainder.
 */
function divide(dividend, divisor) {
    if (divisor == 0) {
        throw new Error('Divisor cannot be zero.');
    }

    if (dividend == 0) {
        return 0;
    }

    if (dividend < divisor) {
        return 0;
    }

    let quotient = 0;
    while (dividend >= divisor) {
        dividend -= divisor;
        quotient++;
    }

    return quotient;
}

console.log('========= Q39 =========');
let dividend = 20;
let divisor = 5;
console.log(
    `Quotient for ${dividend}/${divisor}: ${divide(dividend, divisor)}`
);

dividend = 30;
divisor = 6;
console.log(
    `Quotient for ${dividend}/${divisor}: ${divide(dividend, divisor)}`
);
console.log('\n');

/*
 * Q40.
 * Determine whether a tree is a valid binary search tree.
 * A binary search tree is a tree with two children, left and right, and
 * satisfies the constraint that the key in the left child must be less than or
 * equal to the root and the key in the right child must be greater than or
 * equal to the root.
 */
class BinarySearchTree {
    constructor() {
        this.root = null;
    }

    insert(key) {
        this.root = this.insertRec(this.root, key);
    }

    insertRec(root, key) {
        if (!root) {
            root = new Node(key);
            return root;
        }

        if (key < root.val) {
            root.left = this.insertRec(root.left, key);
        } else if (key > root.val) {
            root.right = this.insertRec(root.right, key);
        }

        return root;
    }

    delete(key) {
        this.root = this.deleteRec(this.root, key);
    }

    deleteRec(root, key) {
        if (!root) {
            return root;
        }

        if (key < root.val) {
            root.left = this.deleteRec(root.left, key);
        } else if (key > root.val) {
            root.right = this.deleteRec(root.right, key);
        } else {
            if (!root.left) {
                return root.right;
            } else if (!root.right) {
                return root.left;
            }

            root.val = this.minValue(root.right);
            root.right = this.deleteRec(root.right, root.val);
        }

        return root;
    }

    minValue(root) {
        let minv = root.val;
        while (root.left) {
            minv = root.left.val;
            root = root.left;
        }
        return minv;
    }

    search(key) {
        return this.searchRec(this.root, key);
    }

    searchRec(root, key) {
        if (!root || root.val === key) {
            return root !== null;
        }

        if (key < root.val) {
            return this.searchRec(root.left, key);
        }
        return this.searchRec(root.right, key);
    }

    inorderTraversal() {
        this.inorderTraversalRec(this.root);
    }

    inorderTraversalRec(root) {
        if (root) {
            this.inorderTraversalRec(root.left);
            console.log(root.val);
            this.inorderTraversalRec(root.right);
        }
    }
}

console.log('========= Q40 =========');
const bst = new BinarySearchTree();
bst.insert(50);
bst.insert(30);
bst.insert(20);
bst.insert(40);
bst.insert(70);
bst.insert(60);
bst.insert(80);

console.log('Inorder traversal:');
bst.inorderTraversal();

let key = 40;
console.log(`${key} is ${bst.search(key) ? '' : 'not '}present in the BST`);

key = 55;
console.log(`${key} is ${bst.search(key) ? '' : 'not '}present in the BST`);

bst.delete(20);
console.log('Inorder traversal after deleting 20:');
bst.inorderTraversal();
console.log('\n');

/*
 * Q41.
 * Given an integer n and a list of integers l, write a function that randomly
 * generates a number from 0 to n-1 that isn't in l (uniform).
 */
function generateRandomNumber(n, l) {
    let available = [];

    for (let i = 0; i < n; i++) {
        if (!l.includes(i)) {
            available.push(i);
        }
    }
    const index = Math.floor(Math.random() * available.length);

    return available[index];
}

console.log('========= Q41 =========');
const n = 10;
const l = [2, 4, 6, 8];
console.log(`Random number: ${generateRandomNumber(n, l)}`);
console.log('\n');

/*
 * Q42.
 * Write a map implementation with a get function that lets you retrieve the
 * value of a key at a particular time.
 * It should contain the following methods:
 * set(key, value, time): sets key to value for t = time.
 * get(key, time): gets the key at t = time.
 * The map should work like this. If we set a key at a particular time, it will
 * maintain that value forever or until it gets set at a later time. In other
 * words, when we get a key at a time, it should return the value that was set
 * for that key set at the most recent time.
 * Consider the following examples:
 * d.set(1, 1, 0) # set key 1 to value 1 at time 0
 * d.set(1, 2, 2) # set key 1 to value 2 at time 2
 * d.get(1, 1) # get key 1 at time 1 should be 1
 * d.get(1, 3) # get key 1 at time 3 should be 2
 * d.set(1, 1, 5) # set key 1 to value 1 at time 5
 * d.get(1, 0) # get key 1 at time 0 should be null
 * d.get(1, 10) # get key 1 at time 10 should be 1
 * d.set(1, 1, 0) # set key 1 to value 1 at time 0
 * d.set(1, 2, 0) # set key 1 to value 2 at time 0
 * d.get(1, 0) # get key 1 at time 0 should be 2
 */
class TimeMap {
    #_map;

    constructor() {
        this.#_map = new Map();
    }

    set(key, value, time) {
        if (!this.#_map.has(key)) {
            this.#_map.set(key, new Map());
        }
        this.#_map.get(key).set(time, value);
    }

    get(key, time) {
        if (this.#_map.has(key)) {
            const values = this.#_map.get(key);

            let floorTime = null;
            for (const [key, value] of values.entries()) {
                if (key <= time) {
                    floorTime = key;
                } else {
                    break;
                }
            }

            if (floorTime !== null) {
                return values.get(floorTime);
            }
        }
        return null;
    }
}

console.log('========= Q42 =========');
const timeMap1 = new TimeMap();
timeMap1.set(1, 1, 0);
timeMap1.set(1, 2, 2);
console.log(`get(1, 1): ${timeMap1.get(1, 1)}`);
console.log(`get(1, 3): ${timeMap1.get(1, 3)}`);

const timeMap2 = new TimeMap();
timeMap2.set(1, 1, 5);
console.log(`get(1, 0): ${timeMap2.get(1, 0)}`);
console.log(`get(1, 10): ${timeMap2.get(1, 10)}`);

const timeMap3 = new TimeMap();
timeMap3.set(1, 1, 0);
timeMap3.set(1, 2, 0);
console.log(`get(1, 0): ${timeMap3.get(1, 0)}`);
console.log('\n');

/*
 * Q43.
 * Given an unsorted array of integers, find the length of the longest
 * consecutive elements sequence.
 * For example, given [100, 4, 200, 1, 3, 2], the longest consecutive element
 * sequence is [1, 2, 3, 4]. Return its length: 4.
 * Your algorithm should run in O(n) complexity.
 */
function longestConsecutive(nums) {
    const set = new Set(nums);
    let maxLength = 0;

    for (const num of nums) {
        if (!set.has(num - 1)) {
            let currentNum = num;
            let currentLength = 1;

            while (set.has(currentNum + 1)) {
                currentNum++;
                currentLength++;
            }

            maxLength = Math.max(maxLength, currentLength);
        }
    }

    return maxLength;
}

console.log('========= Q43 =========');
const numsToFindLongestConsecutive = [100, 4, 200, 1, 3, 2];
console.log(
    `Longest consecutive sequence length: ${longestConsecutive(
        numsToFindLongestConsecutive
    )}`
);
console.log('\n');

/*
 * Q44.
 * Given a list of integers and a number K, return which contiguous elements of
 * the list sum to K.
 * For example, if the list is [1, 2, 3, 4, 5] and K is 9, then it should return
 * [2, 3, 4], since 2 + 3 + 4 = 9.
 */
function findContiguousElementsSum(nums, k) {
    let result = [];
    let left = 0;
    let right = 0;
    let sum = 0;

    while (right < nums.length) {
        sum += nums[right];

        while (sum > k) {
            sum -= nums[left];
            left++;
        }

        if (sum === k) {
            for (let i = left; i <= right; i++) {
                result.push(nums[i]);
            }
            return result;
        }
        right++;
    }
    return result;
}

console.log('========= Q44 =========');
const numsToFindContiguousSum = [1, 2, 3, 4, 5];
const targetContiguousSum = 9;
console.log(
    `Contiguous elements sum: ${findContiguousElementsSum(
        numsToFindContiguousSum,
        targetContiguousSum
    )}`
);
console.log('\n');

/*
 * Q45.
 * Given a string and a set of characters, return the shortest substring
 * containing all the characters in the set.
 * For example, given the string "figehaeci" and the set of characters {a, e,
 * i}, you should return "aeci".
 * If there is no substring containing all the characters in the set, return
 * null.
 */
function shortestSubstring(s, charSet) {
    let charCounts = new Map();
    for (const c of charSet) {
        charCounts.set(c, charCounts.get(c) + 1 || 1);
    }

    let left = 0;
    let right = 0;
    let count = charSet.size;
    let minLen = Number.MAX_SAFE_INTEGER;
    let minStart = 0;

    while (right < s.length) {
        const c = s.charAt(right);
        if (charCounts.has(c)) {
            charCounts.set(c, charCounts.get(c) - 1);
            if (charCounts.get(c) === 0) {
                count--;
            }
        }

        while (count === 0) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }

            const leftChar = s.charAt(left);
            if (charCounts.has(leftChar)) {
                charCounts.set(leftChar, charCounts.get(leftChar) + 1);
                if (charCounts.get(leftChar) > 0) {
                    count++;
                }
            }
            left++;
        }
        right++;
    }

    if (minLen === Number.MAX_SAFE_INTEGER) {
        return null;
    }
    return s.substring(minStart, minStart + minLen);
}

console.log('========= Q45 =========');
const str = 'figehaeci';
const charSet = new Set(['a', 'e', 'i']);
console.log(`Shortest substring: ${shortestSubstring(str, charSet)}`);
console.log('\n');

/*
 * Q46.
 * Given an integer list where each number represents the number of hops you can
 * make, determine whether you can reach to the last index starting at index 0.
 * For example, [2, 0, 1, 0] returns True while [1, 1, 0, 1] returns False.
 */
function canReachLastIndex(nums) {
    let maxReach = 0;
    const n = nums.length;

    for (let i = 0; i < n; i++) {
        if (i > maxReach) {
            // If the current index is not reachable, return false
            return false;
        }
        maxReach = Math.max(maxReach, i + nums[i]);

        if (maxReach >= n - 1) {
            // If the maximum reachable index is greater than or equal to the last index,
            // we can reach the last index
            return true;
        }
    }
    return false;
}

console.log('========= Q46 =========');
const numArr1 = [2, 0, 1, 0];
console.log(`Can reach last index: ${canReachLastIndex(numArr1)}`);
const numArr2 = [1, 1, 0, 1];
console.log(`Can reach last index: ${canReachLastIndex(numArr2)}`);
console.log('\n');

/*
 * Q47.
 * Given an unsigned 8-bit integer, swap its even and odd bits. The 1st and 2nd
 * bit should be swapped, the 3rd and 4th bit should be swapped, and so on.
 * For example, 10101010 should be 01010101. 11100010 should be 11010001.
 * Bonus: Can you do this in one line?
 */
function swapEvenOddBits(num) {
    return ((num & 0b10101010) >>> 1) | ((num & 0b01010101) << 1);
}

console.log('========= Q47 =========');
const num1 = 0b10101010;
console.log(`Swapped even and odd bits: ${swapEvenOddBits(num1).toString(2)}`);
const num2 = 0b11100010;
console.log(`Swapped even and odd bits: ${swapEvenOddBits(num2).toString(2)}`);
console.log('\n');

/*
 * Q48.
 * Given a binary tree, return all paths from the root to leaves.
 * For example, given the tree:
 * "   1       "
 * "  / \      "
 * " 2   3     "
 * "    / \    "
 * "   4   5   "
 * Return [[1, 2], [1, 3, 4], [1, 3, 5]].
 */
function binaryTreePaths(root) {
    let paths = [];
    let currentPath = [];
    dfsForTreePath(root, currentPath, paths);
    return paths;
}

function dfsForTreePath(node, currentPath, paths) {
    if (!node) {
        return;
    }

    currentPath.push(node.val);

    if (!node.left && !node.right) {
        paths.push(currentPath.slice());
    } else {
        dfsForTreePath(node.left, currentPath, paths);
        dfsForTreePath(node.right, currentPath, paths);
    }

    currentPath.pop();
}

console.log('========= Q48 =========');
let rootToFindAllLeafPaths = new TreeNode(1);
rootToFindAllLeafPaths.left = new TreeNode(2);
rootToFindAllLeafPaths.right = new TreeNode(3);
rootToFindAllLeafPaths.right.left = new TreeNode(4);
rootToFindAllLeafPaths.right.right = new TreeNode(5);

const paths = binaryTreePaths(rootToFindAllLeafPaths);

for (const path of paths) {
    console.log(path);
}
console.log('\n');

/*
 * Q49.
 * Given a string of words delimited by spaces, reverse the words in string. For
 * example, given "hello world here", return "here world hello"
 * Follow-up: given a mutable string representation, can you perform this
 * operation in-place?
 */
function reverseWords(input) {
    const words = input.split(' ');

    let left = 0;
    let right = words.length - 1;

    while (left < right) {
        const temp = words[left];
        words[left] = words[right];
        words[right] = temp;
        left++;
        right--;
    }

    return words.join(' ');
}

console.log('========= Q49 =========');
const input = 'hello world here';
console.log(`Reversed words: ${reverseWords(input)}`);
console.log('\n');

/*
 * Q50.
 * Generate a finite, but an arbitrarily large binary tree quickly in O(1).
 * That is, generate() should return a tree whose size is unbounded but finite.
 */
class TreeGenerator {
    generate() {
        let root = new TreeNode(1);
        root.left = this.createUnboundedNode();
        root.right = this.createUnboundedNode();
        return root;
    }

    createUnboundedNode() {
        return new TreeNode(-1); // Use a special value to represent an unbounded node
    }
}

console.log('========= Q50 =========');
console.log('\n');

/*
 * Q51.
 * Given a set of closed intervals, find the smallest set of numbers that covers
 * all the intervals. If there are multiple smallest sets, return any of them.
 * For example, given the intervals [0, 3], [2, 6], [3, 4], [6, 9], one set of
 * numbers that covers all these intervals is {3, 6}.
 */
class Interval {
    constructor(start, end) {
        this.start = start;
        this.end = end;
    }
}

function findCoveringSet(intervals) {
    intervals.sort((a, b) => a.start - b.start);
    let end = Number.MIN_SAFE_INTEGER;
    let coveringSet = [];

    for (const interval of intervals) {
        if (interval.start > end) {
            coveringSet.push(end);
        }
        end = Math.max(end, interval.end);
    }
    coveringSet.push(end);

    return coveringSet;
}

console.log('========= Q51 =========');
const intervals = [
    new Interval(0, 3),
    new Interval(2, 6),
    new Interval(3, 4),
    new Interval(6, 9),
];
const coveringSet = findCoveringSet(intervals);
console.log(`Covering set: ${coveringSet}`);
console.log('\n');

/*
 * Q52.
 * Implement the singleton pattern with a twist. First, instead of storing one
 * instance, store two instances. And in every even call of getInstance(),
 * return the first instance and in every odd call of getInstance(), return the
 * second instance.
 */
class TwistedSingleton {
    static #_instance1;
    static #_instance2;
    static #_count = 0;

    getInstance() {
        TwistedSingleton.#_count += 1;

        if (TwistedSingleton.#_count % 2 === 0) {
            if (!TwistedSingleton.#_instance1) {
                TwistedSingleton.#_instance1 = new TwistedSingleton();
            }
            return TwistedSingleton.#_instance1;
        } else {
            if (!TwistedSingleton.#_instance2) {
                TwistedSingleton.#_instance2 = new TwistedSingleton();
            }
            return TwistedSingleton.#_instance2;
        }
    }
}

console.log('========= Q52 =========');
const singleton1 = new TwistedSingleton().getInstance();
const singleton2 = new TwistedSingleton().getInstance();
const singleton3 = new TwistedSingleton().getInstance();

console.log(`Singleton 1: ${singleton1}`);
console.log(`Singleton 2: ${singleton1}`);
console.log(`Singleton 3: ${singleton1}`);
console.log('\n');

/*
 * Q53.
 * You are given a 2-d matrix where each cell represents number of coins in that
 * cell. Assuming we start at matrix[0][0], and can only move right or down,
 * find the maximum number of coins you can collect by the bottom right corner.
 * For example, in this matrix
 * 0 3 1 1
 * 2 0 0 4
 * 1 5 3 1
 * The most we can collect is 0 + 2 + 1 + 5 + 3 + 1 = 12 coins.
 */
function getMaxCoins(matrix) {
    if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
        return 0;
    }

    const m = matrix.length;
    const n = matrix[0].length;

    let dp = new Array(m).fill(0).map(() => new Array(n).fill(0));

    dp[0][0] = matrix[0][0];
    for (let i = 1; i < m; i++) {
        dp[i][0] = dp[i - 1][0] + matrix[i][0];
    }
    for (let j = 1; j < n; j++) {
        dp[0][j] = dp[0][j - 1] + matrix[0][j];
    }

    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
        }
    }

    return dp[m - 1][n - 1];
}

console.log('========= Q53 =========');
const coinMatrix = [
    [0, 3, 1, 1],
    [2, 0, 0, 4],
    [1, 5, 3, 1],
];
const maxCoins = getMaxCoins(coinMatrix);
console.log(`Maximum number of coins collected: ${maxCoins}`);
console.log('\n');

/*
 * Q54.
 * Write a function that rotates a list by k elements. For example, [1, 2, 3, 4,
 * 5, 6] rotated by two becomes [3, 4, 5, 6, 1, 2]. Try solving this without
 * creating a copy of the list. How many swap or move operations do you need?
 */
function rotateList(list, k) {
    const n = list.length;
    k = k % n;

    reverse(list, 0, n - 1);
    reverse(list, 0, n - k - 1);
    reverse(list, n - k, n - 1);
}

function reverse(list, start, end) {
    while (start < end) {
        let temp = list[start];
        list[start] = list[end];
        list[end] = temp;
        start++;
        end--;
    }
}

console.log('========= Q54 =========');
const list = [1, 2, 3, 4, 5, 6];
rotateList(list, 2);
console.log(`Rotated list: ${list}`);
console.log('\n');

/*
 * Q55.
 * The Tower of Hanoi is a puzzle game with three rods and n disks, each a
 * different size.
 * All the disks start off on the first rod in a stack. They are ordered by
 * size, with the largest disk on the bottom and the smallest one at the top.
 * The goal of this puzzle is to move all the disks from the first rod to the
 * last rod while following these rules:
 * You can only move one disk at a time.
 * A move consists of taking the uppermost disk from one of the stacks and
 * placing it on top of another stack.
 * You cannot place a larger disk on top of a smaller disk.
 * Write a function that prints out all the steps necessary to complete the
 * Tower of Hanoi. You should assume that the rods are numbered, with the first
 * rod being 1, the second (auxiliary) rod being 2, and the last (goal) rod
 * being 3.
 * For example, with n = 3, we can do this in 7 moves:
 * Move 1 to 3
 * Move 1 to 2
 * Move 3 to 2
 * Move 1 to 3
 * Move 2 to 1
 * Move 2 to 3
 * Move 1 to 3
 */
function solveTowerOfHanoi(n, source, auxiliary, destination) {
    if (n === 1) {
        console.log(`Move disk 1 from rod ${source} to rod ${destination}`);
        return;
    }

    solveTowerOfHanoi(n - 1, source, destination, auxiliary);
    console.log(`Move disk ${n} from rod ${source} to rod ${destination}`);
    solveTowerOfHanoi(n - 1, auxiliary, source, destination);
}

console.log('========= Q55 =========');
const disks = 3;
const sourceRod = 1;
const auxiliaryRod = 2;
const destinationRod = 3;
solveTowerOfHanoi(disks, sourceRod, auxiliaryRod, destinationRod);
console.log('\n');

/*
 * Q56.
 * Given a real number n, find the square root of n. For example, given n = 9,
 * return 3.
 */
function findSquareRoot(n) {
    if (n < 0) {
        throw new Error('Cannot calculate square root of a negative number');
    }

    if (n === 0) {
        return 0;
    }

    let x = n;
    let y = 0;

    while (x != y) {
        y = x;
        x = (n / x + x) / 2;
    }

    return x;
}

console.log('========= Q56 =========');
const nForSquareRoot = 9;
const squareRoot = findSquareRoot(nForSquareRoot);
console.log(`Square root of ${nForSquareRoot}: ${squareRoot}`);
console.log('\n');

/*
 * Q57.
 * Given an array of numbers representing the stock prices of a company in
 * chronological order and an integer k, return the maximum profit you can make
 * from k buys and sells. You must buy the stock before you can sell it, and you
 * must sell the stock before you can buy it again.
 * For example, given k = 2 and the array [5, 2, 4, 0, 1], you should return 3.
 */
function getMaxProfit(prices, k) {
    const n = prices.length;
    let dp = new Array(k + 1).fill(0).map(() => new Array(n).fill(0));

    for (let i = 1; i <= k; i++) {
        let maxDiff = -prices[0];

        for (let j = 1; j < n; j++) {
            dp[i][j] = Math.max(dp[i][j - 1], prices[j] + maxDiff);
            maxDiff = Math.max(maxDiff, dp[i - 1][j] - prices[j]);
        }
    }
    return dp[k][n - 1];
}

console.log('========= Q57 =========');
const prices = [5, 2, 4, 0, 1];
const numOfBuy = 2;
const maxProfit = getMaxProfit(prices, numOfBuy);
console.log(`Maximum profit: ${maxProfit}`);
console.log('\n');

/*
 * Q58.
 * Given the head to a singly linked list, where each node also has a “random”
 * pointer that points to anywhere in the linked list, deep clone the list.
 */
class SinglyLinkedList {
    constructor(val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}

function cloneLinkedList(head) {
    if (!head) {
        return null;
    }

    let map = new Map();

    // First pass: create cloned nodes without random pointers
    let curr = head;
    while (curr) {
        let clone = new SinglyLinkedList(curr.val);
        map.set(curr, clone);
        curr = curr.next;
    }

    // Second pass: set random pointers for cloned nodes
    curr = head;
    while (curr) {
        let clone = map.get(curr);
        clone.next = map.get(curr.next);
        clone.random = map.get(curr.random);
        curr = curr.next;
    }

    return map.get(head);
}

console.log('========= Q58 =========');
const singlyListHead = new SinglyLinkedList(1);
singlyListHead.next = new SinglyLinkedList(2);
singlyListHead.next.next = new SinglyLinkedList(3);
singlyListHead.next.next.next = new SinglyLinkedList(4);
singlyListHead.next.next.next.next = new SinglyLinkedList(5);

singlyListHead.random = singlyListHead.next.next;
singlyListHead.next.random = singlyListHead;
singlyListHead.next.next.random = singlyListHead.next.next.next.next;
singlyListHead.next.next.next.random = singlyListHead.next.next;
singlyListHead.next.next.next.next.random = singlyListHead.next;

const clonedHead = cloneLinkedList(singlyListHead);

let curr = clonedHead;
while (curr) {
    console.log(`Node value: ${curr.val}`);
    console.log(`Node random value: ${curr.random.val}`);
    curr = curr.next;
}
console.log('\n');

/*
 * Q59.
 * Given a node in a binary search tree, return the next bigger element, also
 * known as the inorder successor.
 * For example, the inorder successor of 22 is 30.
 * "   10          "
 * "  /  \         "
 * " 5    30       "
 * "     /  \      "
 * "   22    35    "
 * You can assume each node has a parent pointer.
 */
function inorderSuccessor(root, target) {
    if (!root) {
        return -1;
    }

    let current = root;
    let successor = null;

    while (current) {
        if (current.val > target) {
            successor = current;
            current = current.left;
        } else {
            current = current.right;
        }
    }

    return successor ? successor.val : -1;
}

console.log('========= Q59 =========');
let bstRootToFindInorderSuccessor = new Node(10);
bstRootToFindInorderSuccessor.left = new Node(5);
bstRootToFindInorderSuccessor.right = new Node(30);
bstRootToFindInorderSuccessor.right.left = new Node(22);
bstRootToFindInorderSuccessor.right.right = new Node(35);

const targetToFindInorderSuccessor = 22;
const successor = inorderSuccessor(
    bstRootToFindInorderSuccessor,
    targetToFindInorderSuccessor
);

if (successor !== -1) {
    console.log(
        `Inorder successor of ${targetToFindInorderSuccessor} is ${successor}`
    );
} else {
    console.log(
        `No inorder successor found for ${targetToFindInorderSuccessor}`
    );
}
console.log('\n');

/*
 * Q60.
 * Given an N by M matrix consisting only of 1's and 0's, find the largest
 * rectangle containing only 1's and return its area.
 * For example, given the following matrix:
 * [[1, 0, 0, 0],
 * [1, 0, 1, 1],
 * [1, 0, 1, 1],
 * [0, 1, 0, 0]]
 * Return 4.
 */
function largestRectangleArea(heights) {
    if (!heights || heights.length === 0) {
        return 0;
    }

    const n = heights.length;
    let maxArea = 0;

    let stack = [];
    for (let i = 0; i <= n; i++) {
        let height = i === n ? 0 : heights[i];

        while (stack.length > 0 && height < heights[stack[stack.length - 1]]) {
            const h = heights[stack.pop()];
            const w = stack.length === 0 ? i : i - stack[stack.length - 1] - 1;
            const area = h * w;
            maxArea = Math.max(maxArea, area);
        }
        stack.push(i);
    }
    return maxArea;
}

function maximalRectangle(matrix) {
    if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
        return 0;
    }

    const rows = matrix.length;
    const cols = matrix[0].length;
    let maxArea = 0;
    let heights = new Array(cols).fill(0);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (matrix[i][j] === 1) {
                heights[j] += 1;
            } else {
                heights[j] = 0;
            }
        }

        const area = largestRectangleArea(heights);
        maxArea = Math.max(maxArea, area);
    }
    return maxArea;
}

console.log('========= Q60 =========');
const rectangleMatrix = [
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
];

const maxArea = maximalRectangle(rectangleMatrix);
console.log(`Maximum rectangle area containing only 1's is: ${maxArea}`);
console.log('\n');

/*
 * Q61.
 * Implement a bit array.
 * A bit array is a space efficient array that holds a value of 1 or 0 at each
 * index.
 * init(size): initialize the array with size
 * set(i, val): updates index at i with val where val is either 1 or 0.
 * get(i): gets the value at index i.
 */
class BitArray {
    #_arr;

    constructor(size) {
        const length = Math.floor((size + 31) / 32);
        this.#_arr = new Array(length);
    }

    set(i, val) {
        if (val !== 0 && val !== 1) {
            throw new Error('Value must be either 0 or 1');
        }

        const index = Math.floor(i / 32); // Calculate the index of the integer
        const bitIndex = i % 32; // Calculate the bit index within the integer

        if (val === 1) {
            this.#_arr[index] |= 1 << bitIndex; // Set the bit at given index
        } else {
            this.#_arr[index] &= ~(1 << bitIndex); // Clear the bit at the given index
        }
    }

    get(i) {
        const index = Math.floor(i / 32); // Calculate the index of the integer
        const bitIndex = i % 32; // Calculate the bit index within the integer
        return (this.#_arr[index] >> bitIndex) & 1; // Get the value of the bit at the given index
    }
}

console.log('========= Q61 =========');
const bitArray = new BitArray(10);
bitArray.set(3, 1);
bitArray.set(7, 1);
bitArray.set(9, 1);

console.log(`Value at index 3: ${bitArray.get(3)}`); // Output: 1
console.log(`Value at index 7: ${bitArray.get(7)}`); // Output: 1
console.log(`Value at index 9: ${bitArray.get(9)}`); // Output: 1
console.log(`Value at index 0: ${bitArray.get(0)}`); // Output: 0
console.log(`Value at index 5: ${bitArray.get(5)}`); // Output: 0
console.log('\n');

/*
 * Q62.
 * Given an iterator with methods next() and hasNext(), create a wrapper
 * iterator, PeekableInterface, which also implements peek(). peek shows the
 * next element that would be returned on next().
 * Here is the interface:
 * "class PeekableInterface(object):   "
 * "    def __init__(self, iterator):  "
 * "        pass                       "
 * "    def peek(self):                "
 * "        pass                       "
 * "    def next(self):                "
 * "        pass                       "
 * "    def hasNext(self):             "
 * "        pass                       "
 */
class PeekableInterface {
    #_iterator;
    #_nextElement;
    #_hasNext;

    constructor(iterator) {
        this.#_iterator = iterator;
        this.#_nextElement = null;
        this.#_hasNext = iterator.hasNext();

        if (this.#_hasNext) {
            this.#_nextElement = iterator.next();
        }
    }

    peek() {
        if (!this.#_hasNext) {
            throw new Error('No more element to peek');
        }
        return this.#_nextElement;
    }

    next() {
        if (!this.#_hasNext) {
            throw new Error('No more elements');
        }

        const currentElement = this.#_nextElement;
        if (this.#_iterator.hasNext()) {
            this.#_nextElement = this.#_iterator.next();
        } else {
            this.#_nextElement = null;
            this.#_hasNext = false;
        }
        return currentElement;
    }

    hasNext() {
        return this.#_hasNext;
    }
}

console.log('========= Q62 =========');
const arr = [1, 2, 3, 4, 5];
const iterator = {
    index: 0,
    next() {
        if (this.index < arr.length) {
            return arr[this.index++];
        }
        return null;
    },
    hasNext() {
        return this.index < arr.length;
    },
};
const peekable = new PeekableInterface(iterator);

console.log(peekable.peek()); // Output: 1
console.log(peekable.next()); // Output: 1
console.log(peekable.next()); // Output: 2
console.log(peekable.peek()); // Output: 3
console.log(peekable.next()); // Output: 3
console.log(peekable.hasNext()); // Output: true
console.log('\n');

/*
 * Q63.
 * Given an array of integers in which two elements appear exactly once and all
 * other elements appear exactly twice, find the two elements that appear only
 * once.
 * For example, given the array [2, 4, 6, 8, 10, 2, 6, 10], return 4 and 8. The
 * order does not matter.
 * Follow-up: Can you do this in linear time and constant space?
 */
function findTwoSingleElements(nums) {
    let xor = 0;

    for (const num of nums) {
        xor ^= num;
    }

    // Find the rightmost set bit indicating a difference between two single elements
    const rightmostSetBit = xor & -xor;

    let num1 = 0;
    let num2 = 0;

    // Divide the numbers into two groups based on the rightmost set bit
    for (const num of nums) {
        if ((num & rightmostSetBit) !== 0) {
            num1 ^= num;
        } else {
            num2 ^= num;
        }
    }

    return [num1, num2];
}

console.log('========= Q63 =========');
const numArr = [2, 4, 6, 8, 10, 2, 6, 10];
const singleAppearances = findTwoSingleElements(numArr);
console.log(`Single elements: ${singleAppearances}`);
console.log('\n');

/*
 * Q64.
 * Given a pivot x, and a list lst, partition the list into three parts.
 * The first part contains all elements in lst that are less than x
 * The second part contains all elements in lst that are equal to x
 * The third part contains all elements in lst that are larger than x
 * Ordering within a part can be arbitrary.
 * For example, given x = 10 and lst = [9, 12, 3, 5, 14, 10, 10], one partition
 * may be [9, 3, 5, 10, 10, 12, 14].
 */
function partitionList(lst, x) {
    let low = 0;
    let high = lst.length - 1;
    let i = 0;

    while (i <= high) {
        if (lst[i] < x) {
            swap(lst, i, low);
            i++;
            low++;
        } else if (lst[i] > x) {
            swap(lst, i, high);
            high--;
        } else {
            i++;
        }
    }
}

console.log('========= Q64 =========');
const pivot = 10;
const lst = [9, 12, 3, 5, 14, 10, 10];

partitionList(lst, pivot);
console.log(`Partitioned list: ${lst}`);
console.log('\n');

/*
 * Q65.
 * Given an array of numbers and an index i, return the index of the nearest
 * larger number of the number at index i, where distance is measured in array
 * indices.
 * For example, given [4, 1, 3, 5, 6] and index 0, you should return 3.
 * If two distances to larger numbers are the equal, then return any one of
 * them. If the array at i doesn't have a nearest larger integer, then return
 * null.
 * Follow-up: If you can preprocess the array, can you do this in constant time?
 */
function findNearestLarger(nums, index) {
    let left = index - 1;
    let right = index + 1;
    const n = nums.length;
    let nearestLargerIndex = null;

    while (left >= 0 || right < n) {
        if (left >= 0 && nums[left] > nums[index]) {
            nearestLargerIndex = left;
            break;
        }
        if (right < n && nums[right] > nums[index]) {
            nearestLargerIndex = right;
            break;
        }
        left--;
        right++;
    }

    return nearestLargerIndex;
}

console.log('========= Q65 =========');
const numsToFindNearestLarger = [4, 1, 3, 5, 6];
const idx = 0;

const nearestLargerIndex = findNearestLarger(numsToFindNearestLarger, idx);
if (nearestLargerIndex !== null) {
    console.log(`Nearest larger number index: ${nearestLargerIndex}`);
} else {
    console.log('No nearest larger number found.');
}
console.log('\n');

/*
 * Q66.
 * Given a binary tree where all nodes are either 0 or 1, prune the tree so that
 * subtrees containing all 0s are removed.
 * For example, given the following tree:
 * "   0       "
 * "  / \      "
 * " 1   0     "
 * "    / \    "
 * "   1   0   "
 * "  / \      "
 * " 0   0     "
 * should be pruned to:
 * "   0       "
 * "  / \      "
 * " 1   0     "
 * "    /      "
 * "   1       "
 * We do not remove the tree at the root or its left child because it still has
 * a 1 as a descendant.
 */
function pruneTree(root) {
    if (!root) {
        return null;
    }

    root.left = pruneTree(root.left);
    root.right = pruneTree(root.right);

    if (!root.left && !root.right && root.val === 0) {
        return null;
    }

    return root;
}

console.log('========= Q66 =========');
const rootToPrune = new TreeNode(0);
rootToPrune.left = new TreeNode(1);
rootToPrune.right = new TreeNode(0);
rootToPrune.right.left = new TreeNode(1);
rootToPrune.right.right = new TreeNode(0);
rootToPrune.right.left.left = new TreeNode(0);
rootToPrune.right.left.right = new TreeNode(0);

const prunedTree = pruneTree(rootToPrune);
printTree(prunedTree);
console.log('\n');

/*
 * Q67.
 * https://en.wikipedia.org/wiki/Gray_code
 * Gray code is a binary code where each successive value differ in only one
 * bit, as well as when wrapping around. Gray code is common in hardware so that
 * we don't see temporary spurious values during transitions.
 * Given a number of bits n, generate a possible gray code for it.
 * For example, for n = 2, one gray code would be [00, 01, 11, 10].
 */
function generateGrayCode(n) {
    if (n <= 0) {
        return [];
    }

    let grayCode = [];
    grayCode.push('0');
    grayCode.push('1');

    for (let i = 2; i <= n; i++) {
        const size = grayCode.length;

        for (let j = size - 1; j >= 0; j--) {
            grayCode.push(grayCode[j]);
        }

        for (let j = 0; j < size; j++) {
            grayCode[j] = '0' + grayCode[j];
            grayCode[j + size] = '1' + grayCode[j + size];
        }
    }
    return grayCode;
}

console.log('========= Q67 =========');
const nForGrayCode = 3;
const grayCode = generateGrayCode(nForGrayCode);
console.log(`Gray code for ${nForGrayCode} bits: ${grayCode}`);
console.log('\n');

/*
 * Q68.
 * Given a 2-D matrix representing an image, a location of a pixel in the screen
 * and a color C, replace the color of the given pixel and all adjacent same
 * colored pixels with C.
 * For example, given the following matrix, and location pixel of (2, 2), and
 * 'G' for green:
 * B B W
 * W W W
 * W W W
 * B B B
 * Becomes
 * B B G
 * G G G
 * G G G
 * B B B
 */
function replaceColour(image, x, y, newColour) {
    const rows = image.length;
    if (rows === 0) {
        return;
    }

    const cols = image[0].length;
    const originalColour = image[x][y];

    if (originalColour === newColour) {
        return;
    }

    replaceColourDFS(image, x, y, originalColour, newColour, rows, cols);
}

function replaceColourDFS(image, x, y, originalColour, newColour, rows, cols) {
    if (
        x < 0 ||
        x >= rows ||
        y < 0 ||
        y >= cols ||
        image[x][y] !== originalColour
    ) {
        return;
    }

    image[x][y] = newColour;

    replaceColourDFS(image, x - 1, y, originalColour, newColour, rows, cols);
    replaceColourDFS(image, x + 1, y, originalColour, newColour, rows, cols);
    replaceColourDFS(image, x, y - 1, originalColour, newColour, rows, cols);
    replaceColourDFS(image, x, y + 1, originalColour, newColour, rows, cols);
}

console.log('========= Q68 =========');
const image = [
    ['B', 'B', 'W'],
    ['W', 'W', 'W'],
    ['W', 'W', 'W'],
    ['B', 'B', 'B'],
];

const pxLocation = [2, 2];
const newColour = 'G';

replaceColour(image, pxLocation[0], pxLocation[1], newColour);

let imageToPrint = '';
for (const row of image) {
    for (const pixel of row) {
        imageToPrint += pixel + ' ';
    }
    imageToPrint += '\n';
}
console.log(imageToPrint);
console.log('\n');

/*
 * Q69.
 * You are given n numbers as well as n probabilities that sum up to 1. Write a
 * function to generate one of the numbers with its corresponding probability.
 * For example, given the numbers [1, 2, 3, 4] and probabilities [0.1, 0.5, 0.2,
 * 0.2], your function should return 1 10% of the time, 2 50% of the time, and 3
 * and 4 20% of the time.
 * You can generate random numbers between 0 and 1 uniformly.
 */
class NumberGenerator {
    #_numbers;
    #_cumulativeProbabilities;

    constructor(numbers, probabilities) {
        if (numbers.length !== probabilities.length) {
            throw new Error(
                'Number of numbers and probabilities must be the same'
            );
        }

        this.#_numbers = numbers;
        this.#_cumulativeProbabilities =
            this.calculateCumulativeProbabilities(probabilities);
    }

    generateNumberWithProbability() {
        const index = this.binarySearch(
            this.#_cumulativeProbabilities,
            Math.random()
        );

        return this.#_numbers[index];
    }

    calculateCumulativeProbabilities(probabilities) {
        const cumulativeProbabilities = new Array(probabilities.length);
        cumulativeProbabilities[0] = probabilities[0];

        for (let i = 1; i < probabilities.length; i++) {
            cumulativeProbabilities[i] =
                cumulativeProbabilities[i - 1] + probabilities[i];
        }
        return cumulativeProbabilities;
    }

    binarySearch(arr, target) {
        let left = 0;
        let right = arr.length - 1;

        while (left < right) {
            const mid = Math.floor((left + right) / 2);

            if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }
}

console.log('========= Q69 =========');
const numbers = [1, 2, 3, 4];
const probabilities = [0.1, 0.5, 0.2, 0.2];
const generator = new NumberGenerator(numbers, probabilities);

for (let i = 0; i < 10; i++) {
    console.log(generator.generateNumberWithProbability());
}
console.log('\n');

/*
 * Q70.
 * Given a list of elements, find the majority element, which appears more than
 * half the time (> floor(len(lst) / 2.0)).
 * You can assume that such element exists.
 * For example, given [1, 2, 1, 1, 3, 4, 0], return 1.
 */
function findMajorityElement(nums) {
    let countMap = new Map();

    for (const num of nums) {
        countMap.set(num, (countMap.get(num) || 0) + 1);
    }

    let majorityElement = 0;
    let majorityCount = 0;

    for (const entry of countMap.entries()) {
        const element = entry[0];
        const count = entry[1];

        if (count > majorityCount) {
            majorityElement = element;
            majorityCount = count;
        }
    }
    return majorityElement;
}

console.log('========= Q70 =========');
const numsForMajorityElement = [1, 2, 1, 1, 3, 4, 0];
console.log(`Majority element: ${findMajorityElement(numsForMajorityElement)}`);
console.log('\n');

/*
 * Q71.
 * Given a positive integer n, find the smallest number of squared integers
 * which sum to n.
 * For example, given n = 13, return 2 since 13 = 3^2 + 2^2 = 9 + 4.
 * Given n = 27, return 3 since 27 = 3^2 + 3^2 + 3^2 = 9 + 9 + 9.
 */
function findSmallestSquaredSum(n) {
    let dp = new Array(n + 1);

    for (let i = 0; i < n; i++) {
        dp[i] = Number.MAX_SAFE_INTEGER;

        for (let j = 1; j * j <= i; j++) {
            // Update dp[i] by considering the minimum of dp[i - j*j] + 1
            dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
        }
    }

    return dp[n];
}

console.log('========= Q71 =========');
let nToFindSquaredInt = 13;
console.log(
    `Smallest number of squared integers for ${nToFindSquaredInt}: ${findSmallestSquaredSum(
        nToFindSquaredInt
    )}`
);

nToFindSquaredInt = 27;
console.log(
    `Smallest number of squared integers for ${nToFindSquaredInt}: ${findSmallestSquaredSum(
        nToFindSquaredInt
    )}`
);
console.log('\n');

/*
 * Q72.
 * You are given an N by M matrix of 0s and 1s. Starting from the top left
 * corner, how many ways are there to reach the bottom right corner?
 * You can only move right and down. 0 represents an empty space while 1
 * represents a wall you cannot walk through.
 * For example, given the following matrix:
 * [[0, 0, 1],
 * [0, 0, 1],
 * [1, 0, 0]]
 * Return two, as there are only two ways to get to the bottom right:
 * Right, down, down, right
 * Down, right, down, right
 * The top left corner and bottom right corner will always be 0.
 */
function countPaths(matrix) {
    const m = matrix.length;
    const n = matrix[0].length;

    let dp = new Array(m).fill(0).map(() => new Array(n).fill(0));

    // Initialize the first row and first column
    dp[0][0] = 1;
    for (let i = 1; i < m; i++) {
        if (matrix[i][0] === 1) {
            break;
        }
        dp[i][0] = 1;
    }
    for (let j = 1; j < n; j++) {
        if (matrix[0][j] === 1) {
            break;
        }
        dp[0][j] = 1;
    }

    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            if (matrix[i][j] === 1) {
                continue;
            }
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}

console.log('========= Q72 =========');
const matrixOf01 = [
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0],
];

console.log(`Number of paths: ${countPaths(matrixOf01)}`);
console.log('\n');

/*
 * Q73.
 * Given a list of words, return the shortest unique prefix of each word. For
 * example, given the list:
 * dog
 * cat
 * apple
 * apricot
 * fish
 * Return the list:
 * d
 * c
 * app
 * apr
 * f
 */
class WordTrieNode {
    #_children;
    #_count;

    constructor() {
        this.#_children = new Map();
        this.#_count = 0;
    }

    get count() {
        return this.#_count;
    }

    set count(value) {
        this.#_count = value;
    }

    get children() {
        return this.#_children;
    }
}

function findShortestUniquePrefix(words) {
    let root = new WordTrieNode();
    buildTrie(root, words);

    let result = [];
    for (const word of words) {
        const prefix = findPrefix(root, word);
        result.push(prefix);
    }

    return result;
}

function buildTrie(root, words) {
    for (const word of words) {
        let current = root;

        for (const c of word) {
            current.count += 1;
            if (!current.children.has(c)) {
                current.children.set(c, new WordTrieNode());
            }
            current = current.children.get(c);
        }
        current.count += 1;
    }
}

function findPrefix(root, word) {
    let prefix = '';
    let current = root;

    for (const c of word) {
        prefix += c;
        current = current.children.get(c);

        if (current.count === 1) {
            break;
        }
    }
    return prefix;
}

console.log('========= Q73 =========');
const wordList = ['dog', 'cat', 'apple', 'apricot', 'fish'];
const prefixes = findShortestUniquePrefix(wordList);

for (const prefix of prefixes) {
    console.log(prefix);
}
console.log('\n');

/*
 * Q74.
 * You are given an array of length n + 1 whose elements belong to the set {1,
 * 2, ..., n}. By the pigeonhole principle, there must be a duplicate. Find it
 * in linear time and space.
 */
function findDuplicate(nums) {
    let slow = nums[0];
    let fast = nums[0];

    // Move slow pointer by one step and fast pointer by two steps
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow !== fast);

    return slow;
}

console.log('========= Q74 =========');
const arrWithDuplicate = [1, 3, 4, 2, 2];
console.log(`Duplicate element: ${findDuplicate(arrWithDuplicate)}`);
console.log('\n');

/*
 * Q75.
 * Given an array of integers, return a new array where each element in the new
 * array is the number of smaller elements to the right of that element in the
 * original input array.
 * For example, given the array [3, 4, 9, 6, 1], return [1, 1, 2, 1, 0], since:
 * There is 1 smaller element to the right of 3
 * There is 1 smaller element to the right of 4
 * There are 2 smaller elements to the right of 9
 * There is 1 smaller element to the right of 6
 * There are no smaller elements to the right of 1
 */
class Element {
    constructor(value, index) {
        this.value = value;
        this.index = index;
    }
}

function countSmallerElements(nums) {
    let counts = new Array(nums.length).fill(0);
    let elements = new Array(nums.length)
        .fill(0)
        .map((_, i) => new Element(nums[i], i));

    mergeSort(elements, 0, nums.length - 1, counts);

    return counts;
}

function mergeSort(elements, start, end, counts) {
    if (start >= end) {
        return;
    }

    let mid = Math.floor((start + end) / 2);
    mergeSort(elements, start, mid, counts);
    mergeSort(elements, mid + 1, end, counts);

    mergeElements(elements, start, mid, end, counts);
}

function mergeElements(elements, start, mid, end, counts) {
    const leftSize = mid - start + 1;
    let leftElements = new Array(leftSize);
    for (let i = 0; i < leftSize; i++) {
        leftElements[i] = elements[start + i];
    }

    const rightSize = end - mid;
    let rightElements = new Array(rightSize);
    for (let i = 0; i < rightSize; i++) {
        rightElements[i] = elements[mid + 1 + i];
    }

    let i = 0,
        j = 0,
        k = start,
        smallerCount = 0;

    while (i < leftSize && j < rightSize) {
        if (leftElements[i].value <= rightElements[j].value) {
            elements[k] = leftElements[i];
            counts[leftElements[i].index] += smallerCount;
            i++;
        } else {
            elements[k] = rightElements[j];
            smallerCount++;
            j++;
        }
        k++;
    }

    while (i < leftSize) {
        elements[k] = leftElements[i];
        counts[leftElements[i].index] += smallerCount;
        i++;
        k++;
    }

    while (j < rightSize) {
        elements[k] = rightElements[j];
        j++;
        k++;
    }
}

console.log('========= Q75 =========');
const numsToFindSmallerRightElements = [3, 4, 9, 6, 1];
console.log(
    `Smaller elements to the right: ${countSmallerElements(
        numsToFindSmallerRightElements
    )}`
);
console.log('\n');

/*
 * Q76.
 * Implement a 2D iterator class. It will be initialized with an array of
 * arrays, and should implement the following methods:
 * next(): returns the next element in the array of arrays. If there are no more
 * elements, raise an exception.
 * has_next(): returns whether or not the iterator still has elements left.
 * For example, given the input [[1, 2], [3], [], [4, 5, 6]], calling next()
 * repeatedly should output 1, 2, 3, 4, 5, 6.
 * Do not use flatten or otherwise clone the arrays. Some of the arrays can be
 * empty.
 */
class TwoDIterator {
    constructor(arrays) {
        this.arrays = arrays;
        this.row = 0;
        this.col = 0;
    }

    hasNext() {
        while (this.row < this.arrays.length) {
            if (this.col < this.arrays[this.row].length) {
                return true;
            }
            this.row++;
            this.col = 0;
        }
        return false;
    }

    next() {
        if (this.hasNext()) {
            const value = this.arrays[this.row][this.col];
            this.col++;
            return value;
        }
        throw new Error('No more elements');
    }
}

console.log('========= Q76 =========');
const arrays = [[1, 2], [3], [], [4, 5, 6]];
const twoDIterator = new TwoDIterator(arrays);

while (twoDIterator.hasNext()) {
    console.log(twoDIterator.next());
}
console.log('\n');

/*
 * Q77.
 * Given an N by N matrix, rotate it by 90 degrees clockwise.
 * For example, given the following matrix:
 * [[1, 2, 3],
 * [4, 5, 6],
 * [7, 8, 9]]
 * you should return:
 * [[7, 4, 1],
 * [8, 5, 2],
 * [9, 6, 3]]
 * Follow-up: What if you couldn't use any extra space?
 */
function rotateMatrix(matrix) {
    const n = matrix.length;

    // Transpose the matrix
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }

    // Reverse each row
    for (let i = 0; i < n; i++) {
        let start = 0;
        let end = n - 1;
        while (start < end) {
            let temp = matrix[i][start];
            matrix[i][start] = matrix[i][end];
            matrix[i][end] = temp;
            start++;
            end--;
        }
    }
}

console.log('========= Q77 =========');
const matrixToRotate = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
];
rotateMatrix(matrixToRotate);
console.log(matrixToRotate);
console.log('\n');

/*
 * Q78.
 * Given a linked list, sort it in O(n log n) time and constant space.
 * For example, the linked list 4 -> 1 -> -3 -> 99 should become -3 -> 1 -> 4 ->
 * 99.
 */
function sortList(head) {
    if (!head || !head.next) {
        return head;
    }

    const middle = getMiddle(head);
    const nextToMiddle = middle.next;

    middle.next = null;

    const left = sortList(head);
    const right = sortList(nextToMiddle);

    const sortedList = mergeLists(left, right);

    return sortedList;
}

function getMiddle(head) {
    if (!head) {
        return null;
    }

    let slow = head;
    let fast = head.next;

    while (fast && fast.next) {
        slow = slow.next;
        fast = fast.next.next;
    }

    return slow;
}

function mergeLists(l1, l2) {
    let dummy = new ListNode(0);
    let curr = dummy;

    while (l1 && l2) {
        if (l1.value <= l2.value) {
            curr.next = l1;
            l1 = l1.next;
        } else {
            curr.next = l2;
            l2 = l2.next;
        }
        curr = curr.next;
    }

    if (l1) {
        curr.next = l1;
    } else if (l2) {
        curr.next = l2;
    }

    return dummy.next;
}

console.log('========= Q78 =========');
let listToSort = new ListNode(4);
listToSort.next = new ListNode(1);
listToSort.next.next = new ListNode(-3);
listToSort.next.next.next = new ListNode(99);

let sortedList = sortList(listToSort);

while (sortedList) {
    console.log(sortedList.value + ' ');
    sortedList = sortedList.next;
}
console.log('\n');

/*
 * Q79.
 * Given a start word, an end word, and a dictionary of valid words, find the
 * shortest transformation sequence from start to end such that only one letter
 * is changed at each step of the sequence, and each transformed word exists in
 * the dictionary. If there is no possible transformation, return null. Each
 * word in the dictionary have the same length as start and end and is
 * lowercase.
 * For example, given start = "dog", end = "cat", and dictionary = {"dot",
 * "dop", "dat", "cat"}, return ["dog", "dot", "dat", "cat"].
 * Given start = "dog", end = "cat", and dictionary = {"dot", "tod", "dat",
 * "dar"}, return null as there is no possible transformation from dog to cat.
 */
function findTransformation(start, end, dictionary) {
    dictionary.push(start);

    //Build the adjacency graph
    const graph = buildGraph(dictionary);

    //Perform BFS traversal
    let parentMap = new Map();
    let queue = [];
    let visited = new Set();

    queue.push(start);
    visited.add(start);

    while (queue.length > 0) {
        const current = queue.shift();

        if (current === end) {
            return constructPath(parentMap, end);
        }

        const transformations = graph.get(current);

        for (const word of transformations) {
            if (!visited.has(word)) {
                parentMap.set(word, current);
                visited.add(word);
                queue.push(word);
            }
        }
    }
    return null;
}

function buildGraph(dictionary) {
    let graph = new Map();

    for (let word of dictionary) {
        graph.set(word, []);

        for (let i = 0; i < word.length; i++) {
            const originalChar = word[i];

            for (let c = 'a'.charCodeAt(0); c <= 'z'.charCodeAt(0); c++) {
                if (c === originalChar.charCodeAt(0)) {
                    continue;
                }

                let transformedWord = '';
                for (let j = 0; j < word.length; j++) {
                    if (i === j) {
                        transformedWord += String.fromCharCode(c);
                    } else {
                        transformedWord += word[j];
                    }
                }

                if (dictionary.includes(transformedWord)) {
                    graph.get(word).push(transformedWord);
                }
            }
            word[i] = originalChar;
        }
    }
    return graph;
}

function constructPath(parentMap, end) {
    let path = [];
    let current = end;

    while (current) {
        path.unshift(current);
        current = parentMap.get(current);
    }

    return path;
}

console.log('========= Q79 =========');
const start = 'dog';
const end = 'cat';
const dictionary1 = ['dot', 'dop', 'dat', 'cat'];
let transformation = findTransformation(start, end, dictionary1);
console.log(
    `${
        transformation
            ? transformation.join(' -> ')
            : 'No transformation sequence found'
    }`
);

const dictionary2 = ['dot', 'tod', 'dat', 'dar'];
transformation = findTransformation(start, end, dictionary2);
console.log(
    `${
        transformation
            ? transformation.join(' -> ')
            : 'No transformation sequence found'
    }`
);
console.log('\n');

/*
 * Q80.
 * Given a string s and a list of words words, where each word is the same
 * length, find all starting indices of substrings in s that is a concatenation
 * of every word in words exactly once.
 * For example, given s = "dogcatcatcodecatdog" and words = ["cat", "dog"],
 * return [0, 13], since "dogcat" starts at index 0 and "catdog" starts at index
 * 13.
 * Given s = "barfoobazbitbyte" and words = ["dog", "cat"], return [] since
 * there are no substrings composed of "dog" and "cat" in s.
 * The order of the indices does not matter.
 */
function findSubstring(s, words) {
    let result = [];
    if (!s || s.length === 0 || !words || words.length === 0) {
        return result;
    }

    const wordLength = words[0].length;
    const totalLength = wordLength * words.length;

    let wordCount = new Map();
    for (const word of words) {
        wordCount.set(word, wordCount.get(word) + 1 || 1);
    }

    for (let i = 0; i <= s.length - totalLength; i++) {
        let currentCount = new Map();
        let j = 0;

        while (j < words.length) {
            const word = s.substring(
                i + j * wordLength,
                i + (j + 1) * wordLength
            );
            if (!wordCount.has(word)) {
                break;
            }

            currentCount.set(word, currentCount.get(word) + 1 || 1);

            if (currentCount.get(word) > wordCount.get(word)) {
                break;
            }
            j++;
        }

        if (j === words.length) {
            result.push(i);
        }
    }
    return result;
}

console.log('========= Q80 =========');
const s1 = 'dogcatcatcodecatdog';
const words1 = ['cat', 'dog'];
console.log(findSubstring(s1, words1));

const s2 = 'barfoobazbitbyte';
const words2 = ['dog', 'cat'];
console.log(findSubstring(s2, words2));
console.log('\n');

/*
 * Q81.
 * Describe and give an example of each of the following types of polymorphism:
 * Ad-hoc polymorphism
 * Parametric polymorphism
 * Subtype polymorphism
 */
// Detailed solution implemented in Java S81
// Ad-hoc polymorphism through function overloading (or operator overloading)
// Parametric polymorphism through generics
// Subtype polymorphism through inheritance and method overriding
console.log('========= Q81 =========');
console.log('\n');

/*
 * Q82.
 * Given the sequence of keys visited by a postorder traversal of a binary
 * search tree, reconstruct the tree.
 * For example, given the sequence 2, 4, 3, 8, 7, 5, you should construct the
 * following tree:
 * "    5      "
 * "   / \     "
 * "  3   7    "
 * " / \   \   "
 * "2   4   8  "
 */
function buildTreeFromPostorderHelper(postorder, start, end) {
    if (start > end) {
        return null;
    }

    const rootVal = postorder[end];
    let root = new TreeNode(rootVal);

    // Find the index of the last element smaller than the root value
    let i = 0;
    for (i = end - 1; i >= start; i--) {
        if (postorder[i] < rootVal) {
            break;
        }
    }

    root.left = buildTreeFromPostorderHelper(postorder, start, i);
    root.right = buildTreeFromPostorderHelper(postorder, i + 1, end - 1);

    return root;
}

function buildTreeFromPostorder(postorder) {
    if (!postorder || postorder.length === 0) {
        return null;
    }

    return buildTreeFromPostorderHelper(postorder, 0, postorder.length - 1);
}

function inorderTraversal(node) {
    if (!node) {
        return;
    }

    inorderTraversal(node.left);
    console.log(node.val);
    inorderTraversal(node.right);
}

console.log('========= Q82 =========');
const postorder = [2, 4, 3, 8, 7, 5];
const treeRootFromPostorder = buildTreeFromPostorder(postorder);
inorderTraversal(treeRootFromPostorder);
console.log('\n');

/*
 * Q83.
 * Given a stack of N elements, interleave the first half of the stack with the
 * second half reversed using only one other queue. This should be done
 * in-place.
 * Recall that you can only push or pop from a stack, and enqueue or dequeue
 * from a queue.
 * For example, if the stack is [1, 2, 3, 4, 5], it should become [1, 5, 2, 4,
 * 3]. If the stack is [1, 2, 3, 4], it should become [1, 4, 2, 3].
 * Hint: Try working backwards from the end state.
 */
function interleaveStack(stack) {
    const size = stack.length;
    const halfSize = Math.floor(size / 2);

    let queue = [];
    let tempStack = new Stack();

    for (let i = 0; i < halfSize; i++) {
        queue.push(stack.pop());
    }

    while (stack.length > 0) {
        tempStack.push(stack.pop());
    }

    while (queue.length > 0) {
        stack.push(tempStack.pop());
        stack.push(queue.shift());
    }

    while (!tempStack.isEmpty()) {
        stack.push(tempStack.pop());
    }
}

console.log('========= Q83 =========');
let stack1 = new Stack();
stack1.push(1);
stack1.push(2);
stack1.push(3);
stack1.push(4);
stack1.push(5);
interleaveStack(stack1);
console.log(stack1.elements);

let stack2 = new Stack();
stack2.push(1);
stack2.push(2);
stack2.push(3);
stack2.push(4);
interleaveStack(stack2);
console.log(stack2.elements);
console.log('\n');

/*
 * Q84.
 * A graph is minimally-connected if it is connected and there is no edge that
 * can be removed while still leaving the graph connected. For example, any
 * binary tree is minimally-connected.
 * Given an undirected graph, check if the graph is minimally-connected. You can
 * choose to represent the graph as either an adjacency matrix or adjacency
 * list.
 */
class Graph {
    #_numVertices;
    #_adjList;

    constructor(numVertices) {
        this.#_numVertices = numVertices;
        this.#_adjList = new Array(numVertices).fill().map(() => []);
    }

    addEdge(u, v) {
        this.#_adjList[u].push(v);
        this.#_adjList[v].push(u);
    }

    isMinimallyConnected() {
        let visited = new Array(this.#_numVertices).fill(false);

        return this.dfs(0, visited, -1) && this.allVisited(visited);
    }

    dfs(vertex, visited, parent) {
        visited[vertex] = true;

        for (let neighbor of this.#_adjList[vertex]) {
            if (!visited[neighbor]) {
                if (!this.dfs(neighbor, visited, vertex)) {
                    return false;
                }
            } else if (neighbor !== parent) {
                return false;
            }
        }
        return true;
    }

    allVisited(visited) {
        for (const v of visited) {
            if (!v) {
                return false;
            }
        }
        return true;
    }

    // S90.
    isBipartite() {
        let colors = new Array(this.#_numVertices).fill(-1);

        for (let i = 0; i < this.#_numVertices; i++) {
            if (colors[i] === -1) {
                if (!this.isBipartiteUtil(i, colors)) {
                    return false;
                }
            }
        }
        return true;
    }

    isBipartiteUtil(src, colors) {
        let queue = [];
        queue.push(src);
        colors[src] = 1;

        while (queue.length > 0) {
            const curr = queue.shift();

            for (const neighbor of this.#_adjList[curr]) {
                if (colors[neighbor] === -1) {
                    colors[neighbor] = 1 - colors[curr];
                    queue.push(neighbor);
                } else if (colors[neighbor] === colors[curr]) {
                    return false;
                }
            }
        }
        return true;
    }
}

console.log('========= Q84 =========');
let minimalGraph = new Graph(5);
minimalGraph.addEdge(0, 1);
minimalGraph.addEdge(0, 2);
minimalGraph.addEdge(2, 3);
minimalGraph.addEdge(2, 4);
console.log(
    `Is the graph minimally-connected? ${minimalGraph.isMinimallyConnected()}`
);
console.log('\n');

/*
 * Q85.
 * What will this code print out?
 * "def make_functions():          "
 * "    flist = []                 "
 * "                               "
 * "    for i in [1, 2, 3]:        "
 * "        def print_i():         "
 * "            print(i)           "
 * "        flist.append(print_i)  "
 * "                               "
 * "    return flist               "
 * "                               "
 * "functions = make_functions()   "
 * "for f in functions:            "
 * "    f()                        "
 * How can we make it print out what we apparently want?
 */
console.log('========= Q85 =========');

/*
 * It will print '3' three times since make_functions() refers to 'i' in its
 * enclosing scope. It should be corrected by using default argument x=i in
 * print_i as below.
 * "def make_functions():          "
 * "    flist = []                 "
 * "                               "
 * "    for i in [1, 2, 3]:        "
 * "        def print_i(x=i):      "
 * "            print(i)           "
 * "        flist.append(print_i)  "
 * "                               "
 * "    return flist               "
 * "                               "
 * "functions = make_functions()   "
 * "for f in functions:            "
 * "    f()                        "
 */
console.log('\n');

/*
 * Q86.
 * Given a circular array, compute its maximum subarray sum in O(n) time. A
 * subarray can be empty, and in this case the sum is 0.
 * For example, given [8, -1, 3, 4], return 15 as we choose the numbers 3, 4,
 * and 8 where the 8 is obtained from wrapping around.
 * Given [-4, 5, 1, 0], return 6 as we choose the numbers 5 and 1.
 */
function getMaxSubarraySum(nums) {
    let maxSum = nums[0];
    let currentMax = nums[0];
    let minSum = nums[0];
    let currentMin = nums[0];
    let totalSum = nums[0];

    for (let i = 1; i < nums.length; i++) {
        totalSum += nums[i];

        currentMax = Math.max(currentMax + nums[i], nums[i]);
        maxSum = Math.max(maxSum, currentMax);

        currentMin = Math.min(currentMin + nums[i], nums[i]);
        minSum = Math.min(minSum, currentMin);
    }

    // If the total sum equals the minimum subarray sum,
    // it means all elements in the array are negative, so return the maximum
    // subarray sum
    if (totalSum == minSum) {
        return maxSum;
    }

    // Otherwise, return the maximum of the maximum subarray sum and the difference
    // between the total sum and the minimum subarray sum
    return Math.max(maxSum, totalSum - minSum);
}

console.log('========= Q86 =========');
const circularArray1 = [8, -1, 3, 4];
console.log(
    `Max subarray sum of ${circularArray1} is ${getMaxSubarraySum(
        circularArray1
    )}`
);

const circularArray2 = [-4, 5, 1, 0];
console.log(
    `Max subarray sum of ${circularArray2} is ${getMaxSubarraySum(
        circularArray2
    )}`
);
console.log('\n');

/*
 * Q87.
 * You are given an array of nonnegative integers. Let's say you start at the
 * beginning of the array and are trying to advance to the end. You can advance
 * at most, the number of steps that you're currently on. Determine whether you
 * can get to the end of the array.
 * For example, given the array [1, 3, 1, 2, 0, 1], we can go from indices 0 ->
 * 1 -> 3 -> 5, so return true.
 * Given the array [1, 2, 1, 0, 0], we can't reach the end, so return false.
 */
function canReachEnd(nums) {
    let lastReachableIndex = nums.length - 1;

    for (let i = nums.length - 2; i >= 0; i--) {
        if (i + nums[i] >= lastReachableIndex) {
            lastReachableIndex = i;
        }
    }

    return lastReachableIndex === 0;
}

console.log('========= Q87 =========');
const stepsArr1 = [1, 3, 1, 2, 0, 1];
console.log(`Can reach end of ${stepsArr1}? ${canReachEnd(stepsArr1)}`);

const stepsArr2 = [1, 2, 1, 0, 0];
console.log(`Can reach end of ${stepsArr2}? ${canReachEnd(stepsArr2)}`);
console.log('\n');

/*
 * Q88.
 * Given a set of distinct positive integers, find the largest subset such that
 * every pair of elements in the subset (i, j) satisfies either i % j = 0 or j %
 * i = 0.
 * For example, given the set [3, 5, 10, 20, 21], you should return [5, 10, 20].
 * Given [1, 3, 6, 24], return [1, 3, 6, 24].
 */
function largestDivisibleSubset(nums) {
    if (!nums || nums.length === 0) {
        return [];
    }

    nums.sort((a, b) => a - b);

    const n = nums.length;
    let dp = new Array(n).fill(0); // Stores the size of the largest subset ending at index i
    let prev = new Array(n).fill(0); // Stores the index of the previous element in the subset

    let maxSize = 0;
    let maxIdx = 0;

    for (let i = 0; i < n; i++) {
        dp[i] = 1;
        prev[i] = -1;

        for (let j = 0; j < i; j++) {
            if (nums[i] % nums[j] === 0 && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                prev[i] = j;
            }
        }

        if (dp[i] > maxSize) {
            maxSize = dp[i];
            maxIdx = i;
        }
    }

    let result = [];
    while (maxIdx !== -1) {
        result.push(nums[maxIdx]);
        maxIdx = prev[maxIdx];
    }

    return result.sort((a, b) => a - b);
}

console.log('========= Q88 =========');
const divSubset1 = [3, 5, 10, 20, 21];
console.log(
    `Largest divisible subset of ${divSubset1} is ${largestDivisibleSubset(
        divSubset1
    )}`
);

const divSubset2 = [1, 3, 6, 24];
console.log(
    `Largest divisible subset of ${divSubset2} is ${largestDivisibleSubset(
        divSubset2
    )}`
);
console.log('\n');

/*
 * Q89.
 * Suppose an array sorted in ascending order is rotated at some pivot unknown
 * to you beforehand. Find the minimum element in O(log N) time. You may assume
 * the array does not contain duplicates.
 * For example, given [5, 7, 10, 3, 4], return 3.
 */
function findMin(nums) {
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
    return nums[left];
}

console.log('========= Q89 =========');
const numsToFindPivot = [5, 7, 10, 3, 4];
console.log(`Pivot of ${numsToFindPivot} is ${findMin(numsToFindPivot)}`);
console.log('\n');

/*
 * Q90.
 * Given an undirected graph G, check whether it is bipartite. Recall that a
 * graph is bipartite if its vertices can be divided into two independent sets,
 * U and V, such that no edge connects vertices of the same set.
 */
console.log('========= Q90 =========');
let bipartiteGraph = new Graph(4);
bipartiteGraph.addEdge(0, 1);
bipartiteGraph.addEdge(1, 2);
bipartiteGraph.addEdge(2, 3);
bipartiteGraph.addEdge(3, 0);

console.log(`Is the graph bipartite? ${bipartiteGraph.isBipartite()}`);
console.log('\n');

/*
* Q91.
* Given a linked list of numbers and a pivot k, partition the linked list so
* that all nodes less than k come before nodes greater than or equal to k.
* For example, given the linked list 5 -> 1 -> 8 -> 0 -> 3 and k = 3, the
* solution could be 1 -> 0 -> 5 -> 8 -> 3.
*/
function partition(head, k) {
    let smallerHead = new ListNode(0);
    let smallerTail = smallerHead;
    let greaterHead = new ListNode(0);
    let greaterTail = greaterHead;

    let curr = head;
    while (curr) {
        if (curr.value < k) {
            smallerTail.next = curr;
            smallerTail = smallerTail.next;
        } else {
            greaterTail.next = curr;
            greaterTail = greaterTail.next;
        }

        curr = curr.next;
    }

    greaterTail.next = null;
    smallerTail.next = greaterHead.next;

    return smallerHead.next;
}

console.log('========= Q91 =========');
let headToPartition = new ListNode(5);
headToPartition.next = new ListNode(1);
headToPartition.next.next = new ListNode(8);
headToPartition.next.next.next = new ListNode(0);
headToPartition.next.next.next.next = new ListNode(3);

const kToPartition = 3;
let newList = partition(headToPartition, kToPartition);

let newListPrint = '';
while (newList) {
    newListPrint += `${newList.value} -> `;
    newList = newList.next;
}
newListPrint += 'null';
console.log(`Partitioned list: ${newListPrint}`);
console.log('\n');

/*
* Q92.
* Given a string and a pattern, find the starting indices of all occurrences of
* the pattern in the string. For example, given the string "abracadabra" and
* the pattern "abr", you should return [0, 7].
*/
function findPatternIndices(str, pattern) {
    let indices = [];

    const n = str.length;
    const m = pattern.length;

    for (let i = 0; i <= n - m; i++) {
        let j;
        for (j = 0; j < m; j++) {
            if (str[i + j] !== pattern[j]) {
                break;
            }
        }

        if (j === m) {
            indices.push(i);
        }
    }

    return indices;
}

console.log('========= Q92 =========');
const strToFindPattern = 'abracadabra';
const pattern = 'abr';
console.log(`Indices of pattern occurrence: ${findPatternIndices(strToFindPattern, pattern)}`);
console.log('\n');

/*
* Q93.
* Given a string of digits, generate all possible valid IP address
* combinations.
* IP addresses must follow the format A.B.C.D, where A, B, C, and D are numbers
* between 0 and 255. Zero-prefixed numbers, such as 01 and 065, are not
* allowed, except for 0 itself.
* For example, given "2542540123", you should return ['254.25.40.123',
* '254.254.0.123'].
*/
function generateIPAddresses(s) {
    let result = [];
    backtrack(s, 0, [], result);
    return result;
}

function backtrack(s, index, current, result) {
    if (index === s.length && current.length === 4) {
        result.push(current.join('.'));
    } else if (index < s.length && current.length < 4) {
        // Try different substrings of s starting from the current index
        for (let i = 1; i <= 3 && index + i <= s.length; i++) {
            const segment = s.substring(index, index + i);
            if (isValidSegment(segment)) {
                current.push(segment);
                backtrack(s, index + i, current, result);
                current.pop();
            }
        }
    }
}

function isValidSegment(segment) {
    if (segment.length > 1 && segment[0] === '0') {
        return false;
    }

    const value = parseInt(segment);
    return value >= 0 && value <= 255;
}

console.log('========= Q93 =========');
const idAddressString = '2542540123';
console.log(`Valid IP addresses: ${generateIPAddresses(idAddressString)}`);
console.log('\n');

/*
* Q94.
* The horizontal distance of a binary tree node describes how far left or right
* the node will be when the tree is printed out.
* More rigorously, we can define it as follows:
* The horizontal distance of the root is 0.
* The horizontal distance of a left child is hd(parent) - 1.
* The horizontal distance of a right child is hd(parent) + 1.
* For example, for the following tree, hd(1) = -2, and hd(6) = 0.
* "             5             "
* "          /     \          "
* "        3         7        "
* "      /  \      /   \      "
* "    1     4    6     9     "
* "   /                /      "
* "  0                8       "
* The bottom view of a tree, then, consists of the lowest node at each
* horizontal distance. If there are two nodes at the same depth and horizontal
* distance, either is acceptable.
* For this tree, for example, the bottom view could be [0, 1, 3, 6, 8, 9].
* Given the root to a binary tree, return its bottom view.
*/
class NodWithHorizontalDistance {
    constructor(value) {
        this.value = value;
        this.hd = Number.MIN_SAFE_INTEGER;
        this.left = null;
        this.right = null;
    }
}

function bottomView(root) {
    let result = [];
    if (!root) {
        return result;
    }

    let map = new Map();

    let queue = [];
    root.hd = 0;
    queue.push(root);

    while (queue.length > 0) {
        let node = queue.shift();
        map.set(node.hd, node.value);

        if (node.left) {
            node.left.hd = node.hd - 1;
            queue.push(node.left);
        }

        if (node.right) {
            node.right.hd = node.hd + 1;
            queue.push(node.right);
        }
    }

    for (const value of map.values()) {
        result.push(value);
    }

    return result.sort();
}

console.log('========= Q94 =========');
let treeRootWithHorizontalDistance = new NodWithHorizontalDistance(5);
treeRootWithHorizontalDistance.left = new NodWithHorizontalDistance(3);
treeRootWithHorizontalDistance.right = new NodWithHorizontalDistance(7);
treeRootWithHorizontalDistance.left.left = new NodWithHorizontalDistance(1);
treeRootWithHorizontalDistance.left.right = new NodWithHorizontalDistance(4);
treeRootWithHorizontalDistance.right.left = new NodWithHorizontalDistance(6);
treeRootWithHorizontalDistance.right.right = new NodWithHorizontalDistance(9);
treeRootWithHorizontalDistance.left.left.left = new NodWithHorizontalDistance(0);
treeRootWithHorizontalDistance.right.right.left = new NodWithHorizontalDistance(8);

console.log(`Bottom view of tree: ${bottomView(treeRootWithHorizontalDistance)}`);
console.log('\n');

/*
* Q95.
* https://en.wikipedia.org/wiki/Roman_numerals
* Given a number in Roman numeral format, convert it to decimal.
* The values of Roman numerals are as follows:
* " {              "
* "     'M': 1000, "
* "     'D': 500,  "
* "     'C': 100,  "
* "     'L': 50,   "
* "     'X': 10,   "
* "     'V': 5,    "
* "     'I': 1     "
* " }              "
* In addition, note that the Roman numeral system uses subtractive notation for
* numbers such as IV and XL.
* For the input XIV, for instance, you should return 14.
*/
function romanToDecimal(roman) {
    let symbolValues = createSymbolValueMap();
    let decimal = 0;

    for (let i = 0; i < roman.length; i++) {
        let currentValue = symbolValues.get(roman[i]);

        if (i + 1 < roman.length && symbolValues.get(roman[i + 1]) > currentValue) {
            decimal -= currentValue;
        } else {
            decimal += currentValue;
        }
    }
    return decimal;
}

function createSymbolValueMap() {
    let symbolValues = new Map();
    symbolValues.set('M', 1000);
    symbolValues.set('D', 500);
    symbolValues.set('C', 100);
    symbolValues.set('L', 50);
    symbolValues.set('X', 10);
    symbolValues.set('V', 5);
    symbolValues.set('I', 1);
    return symbolValues;
}

console.log('========= Q95 =========');
const roman = 'XIV';
console.log(`Roman numeral ${roman} is ${romanToDecimal(roman)} in decimal`);
console.log('\n');

/*
* Q96.
* Write an algorithm that computes the reversal of a directed graph. For
* example, if a graph consists of A -> B -> C, it should become A <- B <- C.
*/
class ReversingGraph {
    #_adjacencyList;

    constructor() {
        this.#_adjacencyList = new Map();
    }

    addEdge(source, destination) {
        if (!this.#_adjacencyList.has(source)) {
            this.#_adjacencyList.set(source, []);
        }
        this.#_adjacencyList.get(source).push(destination);
    }

    reverse() {
        let reversedGraph = new ReversingGraph();

        for (const vertex of this.#_adjacencyList.keys()) {
            for (const destination of this.#_adjacencyList.get(vertex)) {
                reversedGraph.addEdge(destination, vertex);
            }
        }
        return reversedGraph;
    }

    printGraph() {
        let stringToPrint = '';
        for (const vertex of this.#_adjacencyList.keys()) {
            stringToPrint += `${vertex} -> `;
            const neighbors = this.#_adjacencyList.get(vertex);
            for (const neighbor of neighbors) {
                stringToPrint += `${neighbor}\n`;
            }
        }
        console.log(stringToPrint);
    }
}

console.log('========= Q96 =========');
let originalGraph = new ReversingGraph();
originalGraph.addEdge('A', 'B');
originalGraph.addEdge('B', 'C');
originalGraph.addEdge('C', 'D');

console.log('Original graph');
originalGraph.printGraph();

let reversedGraph = originalGraph.reverse();
console.log('Reversed graph');
reversedGraph.printGraph();

/*
* Q97.
* In front of you is a row of N coins, with values v1, v1, ..., vn.
* You are asked to play the following game. You and an opponent take turns
* choosing either the first or last coin from the row, removing it from the
* row, and receiving the value of the coin.
* Write a program that returns the maximum amount of money you can win with
* certainty, if you move first, assuming your opponent plays optimally.
*/
function maxMoney(coins) {
    const n = coins.length;

    return play(coins, 0, n - 1);
}

function play(coins, left, right) {
    if (left > right) {
        return 0;
    }

    let pickLeft = coins[left] + Math.min(play(coins, left + 2, right), play(coins, left + 1, right - 1));
    let pickRight = coins[right] + Math.min(play(coins, left + 1, right - 1), play(coins, left, right - 2));

    let maxMoney = Math.max(pickLeft, pickRight);

    return maxMoney;
}

console.log('========= Q97 =========');
const coins = [3, 9, 1, 2];
console.log(`Maximum amount of money: ${maxMoney(coins)}`);
console.log('\n');

/*
* Q98.
* Given an absolute pathname that may have . or .. as part of it, return the
* shortest standardized path.
* For example, given "/usr/bin/../bin/./scripts/../", return "/usr/bin/".
*/
function shortestPath(path) {
    const components = path.split('/');
    let stack = []

    for (const component of components) {
        if (component === '.' || component === '') {
            continue;
        } else if (component === '..') {
            if (stack.length > 0) {
                stack.pop();
            }
        } else {
            stack.push(component);
        }
    }

    let shortestPath = '/';
    for (const dir of stack) {
        shortestPath += `${dir}/`;
    }

    return shortestPath;
}

console.log('========= Q98 =========');
const path = '/usr/bin/../bin/./scripts/../';
console.log(`Shortest standardized path: ${shortestPath(path)}`);
console.log('\n');

function largestNumber(nums) {
    let numStrings = new Array(nums.length);
    for (let i = 0; i < nums.length; i++) {
        numStrings[i] = nums[i].toString();
    }

    numStrings.sort((a, b) => b[0] - a[0]);

    let largestNumber = '';
    for (const numString of numStrings) {
        largestNumber += numString;
    }

    return largestNumber;
}

console.log('========= Q99 =========');
const numsToFormLargest = [10, 7, 76, 415];
console.log(`Largest number formation: ${largestNumber(numsToFormLargest)}`);
console.log('\n');

/*
* Q100.
* https://en.wikipedia.org/wiki/Snakes_and_Ladders
* Snakes and Ladders is a game played on a 10 x 10 board, the goal of which is
* get from square 1 to square 100. On each turn players will roll a six-sided
* die and move forward a number of spaces equal to the result. If they land on
* a square that represents a snake or ladder, they will be transported ahead or
* behind, respectively, to a new square.
* Find the smallest number of turns it takes to play snakes and ladders.
* For convenience, here are the squares representing snakes and ladders, and
* their outcomes:
* snakes = {16: 6, 48: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95:
* 75, 98: 78}
* ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80:
* 100}
*/
class SnakesAndLadders {

    snakesAndLadders(board) {
        const n = board.length;
        let visited = new Array(n + 1).fill(false);

        let queue = [];
        queue.push(1);
        visited[1] = true;

        let turns = 0;

        while (queue.length > 0) {
            const size = queue.length;

            for (let i = 0; i < size; i++) {
                const square = queue.shift();

                if (square === n - 1) {
                    return turns;
                }

                for (let j = 1; j <= 6 && square + j < n; j++) {
                    const next = board[square + j] === -1 ? square + j : board[square + j];

                    if (!visited[next]) {
                        visited[next] = true;
                        queue.push(next);
                    }
                }
            }
            turns++;
        }
        return -1;
    }
}

console.log('========= Q100 =========');
const snakesAndLadders = new SnakesAndLadders();

let board = new Array(101).fill(-1);

let snakes = new Map();
snakes.set(16, 6);
snakes.set(48, 26);
snakes.set(49, 11);
snakes.set(56, 53);
snakes.set(62, 19);
snakes.set(64, 60);
snakes.set(87, 24);
snakes.set(93, 73);
snakes.set(95, 75);
snakes.set(98, 78);

let ladders = new Map();
ladders.set(1, 38);
ladders.set(4, 14);
ladders.set(9, 31);
ladders.set(21, 42);
ladders.set(28, 84);
ladders.set(36, 44);
ladders.set(51, 67);
ladders.set(71, 91);
ladders.set(80, 100);

for (const [key, value] of snakes) {
    board[key] = value;
}

for (const [key, value] of ladders) {
    board[key] = value;
}

console.log(`Minimum turns to win the game: ${snakesAndLadders.snakesAndLadders(board)}`);
console.log('\n');

/*
* Q101.
* You are given N identical eggs and access to a building with k floors. Your
* task is to find the lowest floor that will cause an egg to break, if dropped
* from that floor. Once an egg breaks, it cannot be dropped again. If an egg
* breaks when dropped from the xth floor, you can assume it will also break
* when dropped from any floor greater than x.
* Write an algorithm that finds the minimum number of trial drops it will take,
* in the worst case, to identify this floor.
* For example, if N = 1 and k = 5, we will need to try dropping the egg at
* every floor, beginning with the first, until we reach the fifth floor, so our
* solution will be 5.
*/
function minTrialDrops(eggs, floors) {
    // Create a 2D array to store the minimum number of trial drops for each subproblem
    let dp = new Array(eggs + 1).fill(0).map(() => new Array(floors + 1).fill(0));

    for (let i = 1; i <= eggs; i++) {
        dp[i][0] = 0;
        dp[i][1] = 1;
    }

    // If there is only one egg, we need to try dropping it from every floor
    for (let j = 1; j <= floors; j++) {
        dp[1][j] = j;
    }

    for (let i = 2; i <= eggs; i++) {
        for (let j = 2; j <= floors; j++) {
            dp[i][j] = Number.MAX_SAFE_INTEGER;
            for (let k = 1; k <= j; k++) {
                const drops = 1 + Math.max(dp[i - 1][k - 1], dp[i][j - k]);
                dp[i][j] = Math.min(dp[i][j], drops);
            }
        }
    }

    return dp[eggs][floors];
}

console.log('========= Q101 =========');
const eggs = 1;
const floors = 5;
console.log(`Minimum number of trial drops required is: ${minTrialDrops(eggs, floors)}`);
console.log('\n');

/*
* Q102.
* You are given a list of N points (x1, y1), (x2, y2), ..., (xN, yN)
* representing a polygon. You can assume these points are given in order; that
* is, you can construct the polygon by connecting point 1 to point 2, point 2
* to point 3, and so on, finally looping around to connect point N to point 1.
* Determine if a new point p lies inside this polygon. (If p is on the boundary
* of the polygon, you should return False).
*/
class Point {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

function isInsidePolygon(polygon, p) {
    const n = polygon.length;
    let count = 0;

    for (let i = 0; i < n; i++) {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];

        // Check if the ray intersects with the edge
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) {
            count++;
        }
    }

    // If the number of intersections is odd, point is inside the polygon
    return count % 2 !== 0;
}

console.log('========= Q102 =========');
const polygon = [new Point(0, 0), new Point(0, 5), new Point(5, 5), new Point(5, 0)];
const p = new Point(1, 1);
console.log(`Point is inside polygon: ${isInsidePolygon(polygon, p)}`);
console.log('\n');

/*
* Q103.
* One way to unlock an Android phone is through a pattern of swipes across a
* 1-9 keypad.
* For a pattern to be valid, it must satisfy the following:
* All of its keys must be distinct.
* It must not connect two keys by jumping over a third key, unless that key has
* already been used.
* For example, 4 - 2 - 1 - 7 is a valid pattern, whereas 2 - 1 - 7 is not.
* Find the total number of valid unlock patterns of length N, where 1 <= N <=
* 9.
*/
class UnlockPatternCalculator {
    calculatePatterns(n) {
        let path = new Array(10).fill(0);
        let count = 0;

        // Generate and count the patterns from 4 corners and 4 sides, and 1 center
        count += 4 * this.countPatterns(path, 1, n - 1);
        count += 4 * this.countPatterns(path, 2, n - 1);
        count += this.countPatterns(path, 5, n - 1);

        return count;
    }

    countPatterns(path, curr, remaining) {
        if (remaining === 0) {
            return 1;
        }

        let count = 0;

        for (let i = 1; i <= 9; i++) {
            if (this.canVisit(path, curr, i)) {
                path[i] = 1;
                // Recursively count the patterns starting from the next key
                count += this.countPatterns(path, i, remaining - 1);
                path[i] = 0;
            }
        }

        return count;
    }

    canVisit(path, curr, next) {
        if (path[next] !== 0) {
            return false;
        }

        let currRow = Math.floor((curr - 1) / 3);
        let currCol = (curr - 1) % 3;

        let nextRow = Math.floor((next - 1) / 3);
        let nextCol = (next - 1) % 3;

        // If the next key is on the same row or column as the current key, return true
        if (currRow === nextRow || currCol === nextCol) {
            return true;
        }

        // If the third key between two keys is already visited, return true
        const mid = Math.floor((curr + next) / 2);
        return path[mid] !== 0;
    }
}

console.log('========= Q103 =========');
const unlockPatternCalculator = new UnlockPatternCalculator();
const numOfSwipes = 4;

console.log(`Number of valid unlock patterns for length ${numOfSwipes}: ${unlockPatternCalculator.calculatePatterns(numOfSwipes)}`);
console.log('\n');

/*
* Q104.
* Given an array of numbers N and an integer k, your task is to split N into k
* partitions such that the maximum sum of any partition is minimized. Return
* this sum.
* For example, given N = [5, 1, 2, 7, 3, 4] and k = 3, you should return 8,
* since the optimal partition is [5, 1, 2], [7], [3, 4].
*/
function isValidPartition(nums, k, maxSum) {
    let sum = 0;
    let partitions = 1;

    for (const num of nums) {
        if (sum + num > maxSum) {
            sum = num;
            partitions++;

            if (partitions > k) {
                return false;
            }
        } else {
            sum += num;
        }
    }

    return true;
}

function partitionArray(nums, k) {
    let maxNum = 0;
    let sumNum = 0;

    for (const num of nums) {
        maxNum = Math.max(maxNum, num);
        sumNum += num;
    }

    let left = maxNum
    let right = sumNum;

    while (left < right) {
        const mid = left + Math.floor((right - left) / 2);

        if (isValidPartition(nums, k, mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}

console.log('========= Q104 =========');
const numsToFindMinimizedMaxPartition = [5, 1, 2, 7, 3, 4];
const numPartition = 3;

console.log(`Minimum maximum sum of partitions: ${partitionArray(numsToFindMinimizedMaxPartition, numPartition)}`);
console.log('\n');

/*
* Q105.
* You are given an array of integers, where each element represents the maximum
* number of steps that can be jumped going forward from that element. Write a
* function to return the minimum number of jumps you must take in order to get
* from the start to the end of the array.
* For example, given [6, 2, 4, 0, 5, 1, 1, 4, 2, 9], you should return 2, as
* the optimal solution involves jumping from 6 to 5, and then from 5 to 9.
*/
function minJumps(nums) {
    const n = nums.length;
    if (n <= 1) {
        return 0;
    }

    let jumps = new Array(n).fill(Number.MAX_SAFE_INTEGER);
    jumps[0] = 0;

    let queue = [];
    queue.push(0);

    while (queue.length > 0) {
        let currentIndex = queue.shift();
        let currentJumps = jumps[currentIndex];

        let maxSteps = nums[currentIndex];
        for (let i = 1; i <= maxSteps; i++) {
            const nextIndex = currentIndex + i;
            if (nextIndex >= n) {
                break;
            }

            if (currentJumps + 1 < jumps[nextIndex]) {
                jumps[nextIndex] = currentJumps + 1;
                queue.push(nextIndex);
            }
        }
    }

    return jumps[n - 1];
}

console.log('========= Q105 =========');
const stepsArr = [6, 2, 4, 0, 5, 1, 1, 4, 2, 9];
console.log(`Minimum number of jumps: ${minJumps(stepsArr)}`);
console.log('\n');

/*
* Q106.
* Given a list of words, determine whether the words can be chained to form a
* circle. A word X can be placed in front of another word Y in a circle if the
* last character of X is same as the first character of Y.
* For example, the words ['chair', 'height', 'racket', touch', 'tunic'] can
* form the following circle: chair --> racket --> touch --> height --> tunic
* --> chair.
*/
function canFormCircle(words) {
    if (!words || words.length === 0) {
        return false;
    }

    const n = words.length;
    let visited = new Array(n).fill(false);

    return dfsWords(words, words.length, visited, words[0], words[0]);
}

function dfsWords(words, wordsSize, visited, startWord, currentWord) {
    if (wordsSize === 1 && startWord[0] === currentWord[currentWord.length - 1]) {
        return true;
    }

    for (let i = 0; i < words.length; i++) {
        if (!visited[i] && currentWord[currentWord.length - 1] === words[i][0]) {
            visited[i] = true;
            if (dfsWords(words, wordsSize - 1, visited, startWord, words[i])) {
                return true;
            }
            visited[i] = false;
        }
    }

    return false;
}

console.log('========= Q106 =========');
const wordsToFormCircle = ['chair', 'height', 'racket', 'touch', 'tunic'];
console.log(`Can form circle: ${canFormCircle(wordsToFormCircle)}`);
console.log('\n');

/*
* Q107.
* A cryptarithmetic puzzle is a mathematical game where the digits of some
* numbers are represented by letters. Each letter represents a unique digit.
* For example, a puzzle of the form:
* "   SEND    "
* " + MORE    "
* "--------   "
* " MONEY     "
* may have the solution:
* {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O', 0, 'R': 8, 'Y': 2}
* Given a three-word puzzle like the one above, create an algorithm that finds
* a solution.
*/
function solvePuzzle(word1, word2, result) {
    let letters = new Set();
    for (const letter of word1) {
        letters.add(letter);
    }
    for (const letter of word2) {
        letters.add(letter);
    }
    for (const letter of result) {
        letters.add(letter);
    }

    let assignments = new Map();
    let digits = new Array(10).fill(0);
    const foundSolution = solveRecursively(word1, word2, result, letters, assignments, digits, 0);

    if (foundSolution) {
        return assignments;
    } else {
        return null;
    }
}

function solveRecursively(word1, word2, result, letters, assignments, digits, index) {
    if (index === letters.size) {
        return evaluateExpression(word1, word2, result, assignments);
    }

    let chars = new Array(letters.length);
    let i = 0;
    for (const c of letters) {
        chars[i++] = c;
    }

    for (let digit = (index === 0 ? 1 : 0); digit <= 9; digit++) {
        if (digits[digit] === 0) {
            if (digit === 0 && result[0] === chars[index]) {
                continue; // Skip leading zero assignment if it matches the first character of the result
            }

            assignments.set(chars[index], digit);
            digits[digit] = 1;

            if (solveRecursively(word1, word2, result, letters, assignments, digits, index + 1)) {
                return true;
            }

            digits[digit] = 0;
        }
    }

    return false;
}

function evaluateExpression(word1, word2, result, assignments) {
    const num1 = getNumericValue(word1, assignments);
    const num2 = getNumericValue(word2, assignments);
    const res = getNumericValue(result, assignments);

    return num1 + num2 === res;
}

function getNumericValue(word, assignments) {
    let value = 0;
    for (const c of word) {
        value = value * 10 + assignments.get(c);
    }

    return value;
}

console.log('========= Q107 =========');
const word1 = 'SEND';
const word2 = 'MORE';
const wordsMathResult = 'MONEY';

const solution = solvePuzzle(word1, word2, wordsMathResult);

if (solution) {
    console.log(`Solution: found: `);
    for (const [key, value] of solution) {
        console.log(`${key} = ${value}`);
    }
} else {
    console.log('No solution found');
}
console.log('\n');

/*
* Q108.
* Given an array of a million integers between zero and a billion, out of
* order, how can you efficiently sort it? Assume that you cannot store an array
* of a billion elements in memory.
*/
console.log('========= Q108 =========');
/*
* High-level approach:
* 1. Divide the array into smaller chunks that can fit into memory.
* 2. Sort each chunk individually using an efficient in-memory sorting such as
* quicksort or mergesort.
* 3. Write each sorted chunk to a single file. Structure data in the file using
* priority queue or min-heap.
* 4. Perform a k-way merge sort on the sorted files. Repeatedly select the
* smallest element from each file and write it to the output file.
*/
console.log('\n');

/*
* Q109.
* Given a string and a number of lines k, print the string in zigzag form. In
* zigzag, characters are printed out diagonally from top left to bottom right
* until reaching the kth line, then back up to top right, and so on.
* For example, given the sentence "thisisazigzag" and k = 4, you should print:
* " t     a     g "
* "  h   s z   a  "
* "   i i   i z   "
* "    s     g    "
*/
function printZigzag(s, k) {
    if (k === 1) {
        console.log(s);
        return;
    }

    let rows = new Array(k).fill('');
    let row = 0;
    let goingDown = true;
    let firstCh = true;

    for (const c of s) {
        if (goingDown) {
            for (let i = 0; i < row; i++) {
                rows[row] += ' ';
            }

            rows[row] += c;

            for (let i = 0; i < k - row - 1; i++) {
                rows[row] += ' ';
            }
        } else {
            for (let i = k - row - 1; i > 0; i--) {
                rows[row] += ' ';
            }

            rows[row] += c;

            for (let i = row; i > 0; i--) {
                rows[row] += ' ';
            }
        }

        if (row == 0) {
            goingDown = true;
            if (!firstCh) {
                for (let i = 0; i < k; i++) {
                    rows[row] += ' ';
                }
            }
            firstCh = false;
        } else if (row == k - 1) {
            goingDown = false;
            for (let i = 0; i < k; i++) {
                rows[row] += ' ';
            }
        }

        row += goingDown ? 1 : -1;
    }

    let result = '';
    for (const row of rows) {
        result += `${row}\n`;
    }

    console.log(result);
}

console.log('========= Q109 =========');
const sentence = 'thisisazigzag';
const numLines = 4;
printZigzag(sentence, numLines);
console.log('\n');

/*
* Q110.
* Recall that a full binary tree is one in which each node is either a leaf
* node, or has two children. Given a binary tree, convert it to a full one by
* removing nodes with only one child.
* For example, given the following tree:
* "          0                "
* "       /     \             "
* "     1         2           "
* "   /            \          "
* " 3                 4       "
* "   \             /   \     "
* "     5          6     7    "
* You should convert it to:
* "     0             "
* "   /     \         "
* " 5         4       "
* "         /   \     "
* "        6     7    "
*/
function convertToFullBinaryTree(root) {
    if (!root) {
        return null;
    }

    if (root.left && root.right){
        root.left = convertToFullBinaryTree(root.left);
        root.right = convertToFullBinaryTree(root.right);
    } else if (root.left) {
        root = convertToFullBinaryTree(root.left);
    } else if (root.right) {
        root = convertToFullBinaryTree(root.right);
    }

    return root;
}

console.log('========= Q110 =========');
const treeToConvert = new TreeNode(0);
treeToConvert.left = new TreeNode(1);
treeToConvert.right = new TreeNode(2);
treeToConvert.left.left = new TreeNode(3);
treeToConvert.left.left.right = new TreeNode(5);
treeToConvert.right.right = new TreeNode(4);
treeToConvert.right.right.left = new TreeNode(6);
treeToConvert.right.right.right = new TreeNode(7);

console.log('Original Binary Tree:');
printInorder(treeToConvert);

const fullBinaryTree = convertToFullBinaryTree(treeToConvert);

console.log('Full Binary Tree:');
printInorder(fullBinaryTree);
console.log('\n');
