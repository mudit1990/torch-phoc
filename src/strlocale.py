# coding=utf-8
import unicodedata
import abc
import regex

#    Copyright (c) 2018
#    Jerod Weinman <jerod@acm.org>
#    $Id$

class AbstractLocale(object):
    """General class for adapting character strings to a valid character set."""

    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def represent(self, input_str):
        """Convert a string into the appropriate form 
        (e.g., for use in a lexicon)
        Input: input_str, a unicode string
        Returns a string that is the filtered version of input_str
        """
        pass

    
    @abc.abstractmethod
    def representable(self, input_str):
        """Determine whether it is possible to convert a string into the
        appropriate form (e.g., for use in a lexicon)
        Input: input_str, a unicode string
        Returns: a boolean indicating whether input_str is representable
        """
        pass

    def toUpperCase(self, input_str):
        """Return all possible form of uppercase versions of a string
        Input: input The string to capitalize
        Returns: a list of uppercase versions of input_str
        """
        return [input_str.upper()]

    
class UniversalLocale(AbstractLocale):
    """An identity-transform locale: strings are represented unchanged."""

    def represent(self, input_str):
        return input_str

    def representable(self, input_str):
        return True




class BasicLocale(AbstractLocale):
    """A simple Locale that accepts only alphanumeric elements in
    [A-Za-z0-9]. Characters with diacritics are morphed into their
    non-diacritical counterparts, select ligatures are expanded, and other 
    characters are filtered."""

    # Default list to match maps/read/locale/BasicLocale.java (covering GNIS)
    __filtered = [ unichr(val) for val in
                   [0x2013, # Non-standard English dash
                    0x02BB, # Modified letter turned comma
                    0x2018, # Turned up quote 
                    0x2019, # Turned down quote
                    0x200E, # Left to Right Mark
                    0x001B, # Escape (ESC)
                    0x00BF  # Inverted question mark
                   ] ]

    def __init__(self, filtered=None):
        """Initialize  with a potential list of unicode characters to remove"""
        if filtered is not None:
            self.__filtered = filtered
    
    
    # TODO: Expand all other ligatures per
    # https://en.wikipedia.org/wiki/Typographic_ligature#Ligatures_in_Unicode_(Latin_alphabets)

    # Mapping of diacritics to simplified forms
    _diacrit_str    = u'ÀÁÂÄÅÇÈÉÊÍÎÏÑÒÓÔÖÚÜàáâäåçèéêíïñòóôöúüĀāĆćČĒēĪīŁłŌōŘřŠšŪū'
    _nondiacrit_str = u'AAAAACEEEIIINOOOOUUaaaaaceeeiinoooouuAaCcCEeIiLlOoRrSsUu'

    _diacrit_dict = {}

    for in_char,out_char in zip(_diacrit_str,_nondiacrit_str):
        _diacrit_dict[in_char] = out_char

    def represent(self, input_str):
        """Basic string filtering. Removes all non-alphanumeric ASCII characters
        ([48, 57], [65, 90], [97, 122]) and morphs diacritics into 
        non-diacritical equivalents.
        """
        return ''.join(
            self.__filterMisc(
                self.__morphDiacritics(
                    self.__filterASCII (
                        [c for c in input_str]))) )
                        

    def representable(self, input_str):
        """All strings are representable except those with value
          U+FFFD (65533), 'REPLACEMENT CHARACTER'
        """
        return not unichr(0xFFFD) in input_str
        

    def __filterASCII(self,chars):
        """Remove non-alphanumeric ASCII characters.
        Input: chars, a list of characters
        Output: ascii_chars, a subsequence of chars where non-alphanumeric ASCII
                               characters have been removed
        """
        ascii_chars = []

        # Super-hacky method to match maps/read/locale/BasicLocale.java
        for c in chars:
            if not (( ord(c) <= 127) and not (
                    ( c>=u'A' and c<=u'Z' ) or ( c>=u'a' and c<=u'z') or
                    ( c>=u'0' and c<=u'9' ))):
                ascii_chars.append(c)

        return ascii_chars


    def __morphDiacritics(self,chars):
        """Convert standard diacritics into non-diacritical forms and 
        replace ligatures

        Input: chars, a list of chars
        Output: filt_chars, a modified sequence of characters 
        """

        filt_chars = []
        
        for c in chars:
            if c in self._diacrit_dict: # Replace simple diacritic
                filt_chars.append(self._diacrit_dict[c]) 
            elif c is unichr(0x00E6):
                filt_chars.extend([u'a',u'e']) # Replace ae ligature
            elif (ord(c) >= 0x0300 and ord(c) <= 0x036F ):
                continue   # Dump combining diacritics
            else:
                filt_chars.append(c)
    
        return filt_chars


    def __filterMisc(self,chars):
        """Elminate miscellaneous offending characters"""

        return [c if (c not in self.__filtered) else u'' for c in chars]


class PrefixBasicLocale(BasicLocale):
    """A simple locale that behaves as 'BasicLocale' except the letter
    case of [upper][lower]{1,2}[upper] prefixes are preserved among
    the strings produced by `toUpperCase()` method."""

    __prefix = regex.compile("^[\p{Lu}][\p{Ll}]{1,2}[\p{Lu}]")
    
    def toUpperCase(self, input_str):
        """Return all possible form of uppercase versions of a string
        Input: input The string to capitalize
        Returns: a list of uppercase versions of input_str
        """

        match = self.__prefix.match(input_str)
        
        if not match:
            return [input_str.upper()]
        else:
            span = match.span()
            return [input_str.upper(),
                    match.group() + input_str[match.end():].upper()]
                    
