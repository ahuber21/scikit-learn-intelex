import re
from collections import OrderedDict, defaultdict

enum = OrderedDict()
gdict = OrderedDict()
gdict["enums"] = defaultdict(lambda: defaultdict(lambda: ""))
enum = 0
doc = "FOO"


def enum(**enums):
    return type("Enum", (), enums)


doc_state = enum(none=0, single=1, multi=2, template=3)


class pcontext(object):
    """Parsing context to keep state between lines"""

    def __init__(self, gdict, ignores, header):
        self.gdict = gdict
        self.ignores = ignores
        self.enum = False
        self.curr_class = False
        self.template = False
        self.access = False
        self.header = header
        self.doc = ""
        self.doc_state = doc_state.none
        self.doc_lambda = None


class enum_parser:
    """Parse an enum"""

    def parse(self, elem, ctxt):
        me = re.match(r"\s*enum +(\w+)\s*", elem)
        if me:
            ctxt.enum = me.group(1)
            # we need a deterministic order when generating API
            ctxt.gdict["enums"][ctxt.enum] = OrderedDict()
            return True
        # if found enum Method, extract the enum values
        if ctxt.enum:
            me = re.match(r".*?}.*", elem)
            if me:
                ctxt.enum = False
                return True
            regex = (
                r"^\s*(\w+)(?:\s*=\s*((\(int\))?\w(\w|:|\s|\+)*))?"
                + r"(\s*,)?\s*((/\*|//).*)?$"
            )
            me = re.match(regex, elem)
            if me and not me.group(1).startswith("last"):
                # save the destination for documentation
                ctxt.doc_lambda = lambda: ctxt.gdict["enums"][ctxt.enum][me.group(1)]
                ctxt.gdict["enums"][ctxt.enum][me.group(1)] = [
                    me.group(2) if me.group(2) else "",
                    ctxt.doc,
                ]
                return True
        return False

    def parse_new(self, elem, ctxt):
        me = re.match(r"\s*enum +(\w+)\s*", elem)
        if me:
            ctxt.enum = me.group(1)
            # we need a deterministic order when generating API
            ctxt.gdict["enums"][ctxt.enum] = OrderedDict()
            return True
        # if found enum Method, extract the enum values
        if ctxt.enum:
            me = re.match(r".*?}.*", elem)
            if me:
                ctxt.enum = False
                return True
            regex = (
                # capture group for value name
                r"^\s*(\w+)"
                # capture group for value (different possible formats, 0x1, (1 << 5), etc.)
                + r"(?:\s*=\s*((\(int\))?(\w|:|\s|\+|\(?\d+\s*<<\s*\d+\)?)*))?"
                # comma after the value, plus possible comments
                + r"(\s*,)?\s*((/\*|//).*)?"
                # EOL
                + r"$"
            )
            me = re.match(regex, elem)
            if me and not me.group(1).startswith("last"):
                # save the destination for documentation
                ctxt.doc_lambda = lambda: ctxt.gdict["enums"][ctxt.enum][me.group(1)]
                ctxt.gdict["enums"][ctxt.enum][me.group(1)] = [
                    me.group(2) if me.group(2) else "",
                    ctxt.doc,
                ]
                return True
        return False


test_enum = """
enum ResultToComputeId
{
    predictionResult  = (1 << 0), /*!< Compute the regular prediction */
    shapContributions = (1 << 1), /*!< Compute SHAP contribution values */
    shapInteractions  = (1 << 2)  /*!< Compute SHAP interaction values */
};
"""

ctx = pcontext(gdict, [], "BAR")

p = enum_parser()
for line in test_enum.split("\n"):
    p.parse_new(line, ctx)

print(gdict)
