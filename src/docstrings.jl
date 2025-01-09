@template (FUNCTIONS, METHODS, MACROS) = """
                    $(SIGNATURES)
                    $(DOCSTRING)
                    """

@template (TYPES) = """
                    $(TYPEDEF)
                    $(DOCSTRING)

                    ---
                    ## Fields
                    $(TYPEDFIELDS)
                    """

@template MODULES = """
                    $(DOCSTRING)

                    ---
                    ## Imports
                    $(IMPORTS)

                    ## Exports
                    $(EXPORTS)
                    """