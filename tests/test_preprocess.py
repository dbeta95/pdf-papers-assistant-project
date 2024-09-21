import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(src_path)

from preprocess import get_sentences, get_semantic_chunks, get_sequential_semantic_chunks

test_string = """
This is a test string that's designed to be easily processed by the different methods in order to get unit tests.

The idea is to passed this text to each preprocessor and then use the output to design simple tests.

The tests will be run using pytests.

Notice that since this code is designed for articles, which are expected to be large texts, the present text string cannot be a single short string.

To make the string larger, the following text is introduced:

One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
"""

def test_get_senteces():
    assert get_sentences(test_string, "asdfgh123") == [{'doc_id': 'asdfgh123',
  'chunk': 1,
  'text': "\nThis is a test string that's designed to be easily processed by the different methods in order to get unit tests."},
 {'doc_id': 'asdfgh123',
  'chunk': 2,
  'text': 'The idea is to passed this text to each preprocessor and then use the output to design simple tests.'},
 {'doc_id': 'asdfgh123',
  'chunk': 3,
  'text': 'The tests will be run using pytests.'},
 {'doc_id': 'asdfgh123',
  'chunk': 4,
  'text': 'Notice that since this code is designed for articles, which are expected to be large texts, the present text string cannot be a single short string.'},
 {'doc_id': 'asdfgh123',
  'chunk': 5,
  'text': "To make the string larger, the following text is introduced:\n\nOne of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear."},
 {'doc_id': 'asdfgh123',
  'chunk': 6,
  'text': 'Teachers and coaches implicitly told us the returns were linear.'},
 {'doc_id': 'asdfgh123',
  'chunk': 7,
  'text': '"You get out," I heard a thousand times, "what you put in."'},
 {'doc_id': 'asdfgh123',
  'chunk': 8,
  'text': 'They meant well, but this is rarely true.'},
 {'doc_id': 'asdfgh123',
  'chunk': 9,
  'text': "If your product is only half as good as your competitor's, you don't get half as many customers."},
 {'doc_id': 'asdfgh123',
  'chunk': 10,
  'text': 'You get no customers, and you go out of business.'},
 {'doc_id': 'asdfgh123',
  'chunk': 11,
  'text': "It's obviously true that the returns for performance are superlinear in business."},
 {'doc_id': 'asdfgh123',
  'chunk': 12,
  'text': 'Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true.'},
 {'doc_id': 'asdfgh123',
  'chunk': 13,
  'text': "But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented."},
 {'doc_id': 'asdfgh123',
  'chunk': 14,
  'text': 'We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity.'},
 {'doc_id': 'asdfgh123',
  'chunk': 15,
  'text': 'In all of these, the rich get richer.'},
 {'doc_id': 'asdfgh123', 'chunk': 16, 'text': '[1]'}]
    
def test_get_semantic_chunks():
    assert get_semantic_chunks(test_string, "asdfgh123") == [{'doc_id': 'asdfgh123',
  'chunk': 1,
  'text': "If your product is only half as good as your competitor's, you don't get half as many customers.\nYou get no customers, and you go out of business."},
 {'doc_id': 'asdfgh123',
  'chunk': 2,
  'text': 'Teachers and coaches implicitly told us the returns were linear.'},
 {'doc_id': 'asdfgh123',
  'chunk': 3,
  'text': 'Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true.\nWe see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity.\nIn all of these, the rich get richer.'},
 {'doc_id': 'asdfgh123',
  'chunk': 4,
  'text': "To make the string larger, the following text is introduced:\n\nOne of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear."},
 {'doc_id': 'asdfgh123',
  'chunk': 5,
  'text': "\nThis is a test string that's designed to be easily processed by the different methods in order to get unit tests.\nNotice that since this code is designed for articles, which are expected to be large texts, the present text string cannot be a single short string."},
 {'doc_id': 'asdfgh123', 'chunk': 6, 'text': '[1]'},
 {'doc_id': 'asdfgh123',
  'chunk': 7,
  'text': 'They meant well, but this is rarely true.'},
 {'doc_id': 'asdfgh123',
  'chunk': 8,
  'text': "It's obviously true that the returns for performance are superlinear in business.\nBut superlinear returns for performance are a feature of the world, not an artifact of rules we've invented."},
 {'doc_id': 'asdfgh123',
  'chunk': 9,
  'text': '"You get out," I heard a thousand times, "what you put in."'},
 {'doc_id': 'asdfgh123',
  'chunk': 10,
  'text': 'The idea is to passed this text to each preprocessor and then use the output to design simple tests.\nThe tests will be run using pytests.'}]
    
def test_get_sequential_semantic_chunks():
    assert get_sequential_semantic_chunks(test_string, "asdfgh123") == [{'doc_id': 'asdfgh123',
  'chunk': 1,
  'text': "\nThis is a test string that's designed to be easily processed by the different methods in order to get unit tests.\nThe idea is to passed this text to each preprocessor and then use the output to design simple tests.\nThe tests will be run using pytests.\nNotice that since this code is designed for articles, which are expected to be large texts, the present text string cannot be a single short string.\nTo make the string larger, the following text is introduced:\n\nOne of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear."},
 {'doc_id': 'asdfgh123',
  'chunk': 2,
  'text': 'Teachers and coaches implicitly told us the returns were linear.\n"You get out," I heard a thousand times, "what you put in."\nThey meant well, but this is rarely true.\nIf your product is only half as good as your competitor\'s, you don\'t get half as many customers.\nYou get no customers, and you go out of business.\nIt\'s obviously true that the returns for performance are superlinear in business.\nSome think this is a flaw of capitalism, and that if we changed the rules it would stop being true.\nBut superlinear returns for performance are a feature of the world, not an artifact of rules we\'ve invented.\nWe see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity.\nIn all of these, the rich get richer.\n[1]'}]